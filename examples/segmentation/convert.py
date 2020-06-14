#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Convert TF-TRT deeplab model.

    Copyright (c) 2020 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import sys
import os
import time

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to pb file.', required=True)
    parser.add_argument('--output', help='Output dir.', default='model')
    args = parser.parse_args()

    model_dir = args.output
    if tf.gfile.Exists(model_dir) == False:
        tf.gfile.MkDir(model_dir)

    if tf.gfile.Exists(args.path) == False:
        print('Error: pb file dose note exist!')
        return

    tf.reset_default_graph()
    graph = tf.Graph()

    graph_def = None
    with tf.gfile.GFile(args.path, 'rb') as f:
        graph_def = tf.GraphDef.FromString(f.read())

    output_names = ['SemanticPredictions']

    converter = trt_convert.TrtGraphConverter(
        input_graph_def=graph_def,
        nodes_blacklist=output_names, #output nodes
        max_batch_size=1,
        # is_dynamic_op=False,
        is_dynamic_op=True,
        max_workspace_size_bytes = 1 << 25,
        precision_mode=trt_convert.TrtPrecisionMode.FP16,
        minimum_segment_size=50)
    trt_graph = converter.convert()

    trt_engine_opts = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("trt_engine_opts = {}".format(trt_engine_opts))

    base_name = os.path.splitext(os.path.basename(args.path))[0]
    save_model_file_name = base_name + '_dynamic_fp16.pb'
    with open(os.path.join(model_dir, save_model_file_name), 'wb') as f:
        f.write(trt_graph.SerializeToString())

if __name__ == "__main__":
    main()
