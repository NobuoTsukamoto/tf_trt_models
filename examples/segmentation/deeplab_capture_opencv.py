#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TF-TRT Deeplab with OpenCV.

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


WINDOW_NAME = "Jetson Nano TF-TRT Deeplab(OpenCV)"
INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'


def get_frozen_graph(graph_file):
    """ Read Frozen Graph file from disk.
    
        Args:
            graph_file: File path to tf-trt model path.

    """
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def create_pascal_label_colormap():
    """ Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(colormap, label):
    """ Adds color defined by the dataset colormap to the label.
    Args:
        colormap: A Colormap for visualizing segmentation results.
        label: A 2D array with integer type, storing the segmentation label.
    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        print(label.shape)
        raise ValueError('Expect 2-D input label')

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


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
    parser.add_argument('--input_size', help='Input size(width=height).', default=513, type=int)
    parser.add_argument('--videopath', help="File path of Videofile.", default='')
    parser.add_argument('--output', help="File path of result.", default="")
    parser.add_argument('--no_display', action='store_false')
    args = parser.parse_args()

    # Initialize window.
    if args.no_display != False:
        cv2.namedWindow(
            WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
        )
        cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Initialize colormap
    colormap = create_pascal_label_colormap()

    # Load graph.
    tf.reset_default_graph()
    graph = get_frozen_graph(args.model)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph, name='')

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

    count = 0
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
        if ret == False:
            print('VideoCapture read return false.')
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resize_im = cv2.resize(image, (args.input_size, args.input_size))

        # Run inference.        
        start_tm = time.time()

        seg_map = tf_sess.run(OUTPUT_TENSOR_NAME,
                              feed_dict={INPUT_TENSOR_NAME: resize_im[None, ...]})

        elapsed_ms = (time.time() - start_tm) * 1000

        # display segmantation map
        seg_image = label_to_color_image(colormap, seg_map[0]).astype(np.uint8)
        seg_image = cv2.resize(seg_image, (w, h))
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # Calc fps.
        elapsed_list.append(elapsed_ms)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(elapsed_ms)
        draw_caption(im, (10, 30),  model_name + ' ' + fps_text + avg_text)
        if count >= 100:
            print(fps_text)
            count = 0
        else:
            count += 1

        # Output video file
        if video_writer != None:
            video_writer.write(im)

        # display
        if args.no_display != False:
            cv2.imshow(WINDOW_NAME, im)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    # When everything done, release the window
    cap.release()
    if video_writer != None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
