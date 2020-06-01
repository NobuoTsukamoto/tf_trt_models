#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TF-TRT Object detection with OpenCV.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import sys
import os
import time
import colorsys
import random

import cv2

import numpy as np

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt

WINDOW_NAME = "Jetson Nano TF-TRT Object detection(OpenCV)"

def get_frozen_graph(graph_file):
    """ Read Frozen Graph file from disk.
    
        Args:
            graph_file: File path to tf-trt model path.

    """
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def ReadLabelFile(file_path):
    """ Function to read labels from text files.
    Args:
        file_path: File path to labels.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def random_colors(N):
    """ Random color generator.
    """
    N = N + 1
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
    random.shuffle(colors)
    return colors

def draw_rectangle(image, box, color, thickness=3):
    """ Draws a rectangle.
    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        color: Rectangle color.
        thickness: Thickness of lines.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)

def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.
    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        caption: String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

def main():
    # parse args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='File path of tf-trt model.', required=True)
    parser.add_argument('--label', help='File path of label.', required=True)
    parser.add_argument('--width', help='Input width.', default=640, type=int)
    parser.add_argument('--height', help='Input height.', default=480, type=int)
    parser.add_argument('--videopath', help="File path of Videofile.", default='')
    parser.add_argument("--output", help="File path of result.", default="")
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Read label file and generate random colors.
    random.seed(42)
    labels = ReadLabelFile(args.label) if args.label else None
    last_key = sorted(labels.keys())[len(labels.keys()) - 1]
    colors = random_colors(last_key)

    # Load graph.
    tf.reset_default_graph()

    graph = get_frozen_graph(args.model)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph, name='')

    input_names = ['image_tensor']
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

    # Video capture.
    if args.videopath == "":
        print('Open camera.')
        GST_STR = 'nvarguscamerasrc \
            ! video/x-raw(memory:NVMM), width={0:d}, height={1:d}, format=(string)NV12, framerate=(fraction)30/1 \
            ! nvvidconv flip-method=2 !  video/x-raw, width=(int){2:d}, height=(int){3:d}, format=(string)BGRx \
            ! videoconvert \
            ! appsink drop=true sync=false'.format(1280, 720, args.width, args.height)
        cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
    else:
        print('Open video file: ', args.videopath)
        cap = cv2.VideoCapture(args.videopath)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

    elapsed_list = []

    # Output Video file
    # Define the codec and create VideoWriter object
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    video_writer = None
    if args.output != '' :
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    while(cap.isOpened()):
        ret, frame = cap.read()
        # for i in range(5):
        #     ret, frame = cap.read()
        if ret == False:
            print('VideoCapture read return false.')
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference.        
        start_tm = time.time()
        scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
            tf_input: image[None, ...]
        })
        elapsed_ms = (time.time() - start_tm) * 1000

        boxes = boxes[0] # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = num_detections[0]

        # plot boxes exceeding score threshold
        for i in range(int(num_detections)):
            if scores[i] >= 0.5:
                # scale box to image coordinates
                box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

                # display rectangle
                box = (box[1], box[0], box[3], box[2])
                draw_rectangle(frame, box, colors[int(classes[i]) - 1])

                # display class name and score
                caption = "{0}({1:.2f})".format(labels[int(classes[i]) - 1], scores[i])
                draw_caption(frame, box, caption)

        # Calc fps.
        elapsed_list.append(elapsed_ms)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(elapsed_ms)
        draw_caption(frame, (10, 30),  model_name + ' ' + fps_text + avg_text)

        # Output video file
        if video_writer != None:
            video_writer.write(frame)

        # display
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer != None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
