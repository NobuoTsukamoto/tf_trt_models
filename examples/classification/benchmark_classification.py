#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TF-TRT Image classification benchmark.

    Copyright (c) 2019 Nobuo Tsukamoto

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
    parser.add_argument('--count', help='Repeat count.', default=100, type=int)
    args = parser.parse_args()

    tf.reset_default_graph()

    graph = get_frozen_graph(args.model)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph, name='')

    input_names = ['input']
    output_names = ['scores']
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_output = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')

    image = Image.open(args.image)

    width = int(tf_input.shape.as_list()[1])
    height = int(tf_input.shape.as_list()[2])

    image = np.array(image.resize((width, height)))

    times = []
    for i in range(arg.count):
        start_tm = time.time()
        tf_sess.run(tf_output, feed_dict={tf_input: image[None, ...]})
        times.append(time.time() - start_tm)
    
    print('Inference : {0:.2f} ms'.format(np.array(times).mean() * 1000))        