#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Convert TF-TRT object detection model.

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
    parser.add_argument('--model', help='tf-trt model.')
    parser.add_argument('--path', help='path to checkpoint dir.')
    parser.add_argument('--output', help='Output dir.', default='model')
    parser.add_argument('--threshold', help='Score threshold', default=0.5, type=float)
    args = parser.parse_args()

    model_dir = args.output
    if tf.gfile.Exists(model_dir) == False:
        tf.gfile.MkDir(model_dir)

    if args.model:
        config_path, checkpoint_path = download_detection_model(args.model, 'data')

    elif args.path:
        if tf.gfile.Exists(args.path) == False:
            print('Error: Checkpoint dir dose note exist!')
            return

        config_path = os.path.join(args.path, 'pipeline.config')
        checkpoint_path = os.path.join(args.path,'model.ckpt')

    else:
        print('Error: Either model or path is not specified in the argument.')
        return

    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        # force_nms_cpu=True,
        force_nms_cpu=False,
        checkpoint=checkpoint_path,
        batch_size=1
    )
    print(input_names, output_names)
    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    save_model_file_name = base_name + '_frozen.pb'
    with open(os.path.join(model_dir, save_model_file_name), 'wb') as f:
        f.write(frozen_graph.SerializeToString())

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
