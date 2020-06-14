#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TF-TRT deeplab benchmark.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import sys
import os
import time

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt

import numpy as np

from PIL import Image

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='File path of tf-trt model.', required=True)
    parser.add_argument('--image', help='File path of image file.', required=True)
    parser.add_argument('--input_size', help='Input size(width = height).', default=513, type=int)
    parser.add_argument('--count', help='Repeat count.', default=100, type=int)
    args = parser.parse_args()

    tf.reset_default_graph()

    graph = get_frozen_graph(args.model)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph, name='')

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    image = Image.open(args.image)
    width, height = image.size
    resize_ratio = 1.0 * args.input_size / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    times = []
    for i in range(args.count + 1):
        start_tm = time.time()

        batch_seg_map = tf_sess.run(OUTPUT_TENSOR_NAME,
                                    feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

        end_tm = time.time()

        if i > 0:
            times.append(end_tm - start_tm)
        else:
            print('First Inference : {0:.2f} ms'.format((end_tm - start_tm)* 1000))

    print('Inference : {0:.2f} ms'.format(np.array(times).mean() * 1000))

if __name__ == "__main__":
    main()
