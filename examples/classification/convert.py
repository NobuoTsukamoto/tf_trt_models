#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Convert TF-TRT Image classification model.

    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import sys
import os
import urllib
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt
# import tensorflow.contrib.tensorrt as trt

from tf_trt_models.classification import download_classification_checkpoint, build_classification_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='tf-trt model.', required=True)
    args = parser.parse_args()

    checkpoint_path, num_classes = download_classification_checkpoint(args.model, 'data')
    frozen_graph, input_names, output_names = build_classification_graph(
        model=args.model,
        checkpoint=checkpoint_path,
        num_classes=num_classes,
        is_remove_relu6=True)
    print(input_names, output_names, num_classes)

    converter = trt.TrtGraphConverter(
        input_graph_def=frozen_graph,
        nodes_blacklist=output_names, #output nodes
        max_batch_size=1,
        is_dynamic_op=False,
        max_workspace_size_bytes = 1 << 25,
        precision_mode=trt.TrtPrecisionMode.FP16, # trt.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES
        minimum_segment_size=50)
    trt_graph = converter.convert()
    # trt_graph = trt.create_inference_graph(
    #     input_graph_def=frozen_graph,
    #     outputs=output_names,
    #     max_batch_size=1,
    #     max_workspace_size_bytes=1 << 25,
    #     precision_mode='FP16',
    #     minimum_segment_size=3
    # )

    trt_engine_opts = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("trt_engine_opts = {}".format(trt_engine_opts))

    model_dir = os.path.join('.', 'model')
    if tf.gfile.Exists(model_dir) == False:
        tf.gfile.MkDir(model_dir)

    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    save_model_file_name = base_name + '_frozen_fp16.pb'
    with open(os.path.join(model_dir, save_model_file_name), 'wb') as f:
        f.write(trt_graph.SerializeToString())

if __name__ == "__main__":
    main()
