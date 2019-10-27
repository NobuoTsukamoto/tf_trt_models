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
import tensorflow.contrib.slim as slim
import numpy as np

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt
# import tensorflow.contrib.tensorrt as trt

from tf_trt_models.detection import download_detection_model, build_detection_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='tf-trt model.', required=True)
    parser.add_argument('--threshold', help='Score threshold', default=0.5, type=float)
    args = parser.parse_args()

    model_dir = os.path.join('.', 'model')
    if tf.gfile.Exists(model_dir) == False:
        tf.gfile.MkDir(model_dir)

    config_path, checkpoint_path = download_detection_model(args.model, 'data')

    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path,
        score_threshold=args.threshold,
        batch_size=1
    )
    print(input_names, output_names)

    # base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    # save_model_file_name = base_name + '_frozen.pb'
    # with open(os.path.join(model_dir, save_model_file_name), 'wb') as f:
    #     f.write(frozen_graph.SerializeToString())

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

    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    save_model_file_name = base_name + '_frozen_fp16.pb'
    with open(os.path.join(model_dir, save_model_file_name), 'wb') as f:
        f.write(trt_graph.SerializeToString())

if __name__ == "__main__":
    main()
