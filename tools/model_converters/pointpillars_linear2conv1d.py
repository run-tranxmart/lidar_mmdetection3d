# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import pdb
import tempfile
import torch
from mmengine import Config
from mmengine.registry import (init_default_scope, TRANSFORMS)
from mmengine.runner import load_state_dict

from mmdet3d.registry import DATASETS, MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='PointPillars model: Linear -> Conv1d')
    parser.add_argument(
        '--src_checkpoint',
        type=str,
        default=None,
        help='Source path of checkpoint')
    parser.add_argument(
        '--des_checkpoint',
        type=str,
        default=None,
        help='Destination path of checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument(
        '--config', type=str, default=None, help='Config path of source model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if (args.src_checkpoint is None) or (not osp.exists(args.src_checkpoint)):
        raise FileExistsError(f'Cannot find checkpoint: {args.src_checkpoint}')
    if (args.config is None) or (not osp.exists(args.config)):
        raise FileExistsError(f'Cannot find config: {args.config}')

    checkpoint = torch.load(args.src_checkpoint)
    src_cfg = Config.fromfile(args.config)

    # Build the model and load checkpoint
    src_cfg.model.train_cfg = None
    init_default_scope(src_cfg.get('default_scope', 'mmdet3d'))
    orig_ckpt = checkpoint['state_dict']
    converted_ckpt = orig_ckpt.copy()
    converted_cfg = src_cfg.copy()

    linear_weight_key = 'voxel_encoder.pfn_layers.0.linear.weight'
    linear_bias_key = 'voxel_encoder.pfn_layers.0.linear.bias'
    conv1d_weight_key = 'voxel_encoder.pfn_layers.0.matmul.weight'
    conv1d_bias_key = 'voxel_encoder.pfn_layers.0.matmul.bias'
    DEL_KEYS = [linear_weight_key, linear_bias_key]
    linear_weight, linear_bias = None, None
    conv1d_weight, conv1d_bias = None, None
    if linear_weight_key in orig_ckpt:
        linear_weight = orig_ckpt[linear_weight_key]
        conv1d_weight = linear_weight.unsqueeze(-1)
    else:
        raise KeyError(f'Cannot find key {linear_weight_key} in checkpoint')

    if linear_bias in orig_ckpt:
        linear_bias = orig_ckpt[linear_bias_key]
        conv1d_bias = linear_bias
    else:
        conv1d_channels = conv1d_weight.size(0)
        conv1d_bias = torch.zeros(conv1d_channels, device=conv1d_weight.device)

    # remove the keys of linear weights and bias
    for key_name in DEL_KEYS:
        if key_name in converted_ckpt:
            converted_ckpt.pop(key_name)

    # add the keys of conv1d weights and bias
    converted_ckpt[conv1d_weight_key] = conv1d_weight
    converted_ckpt[conv1d_bias_key] = conv1d_bias
    converted_cfg.model.voxel_encoder.use_conv1d = True

    # Check the converted checkpoint by loading to the model
    des_model = MODELS.build(converted_cfg.model).to(args.device)
    # pdb.set_trace()
    load_state_dict(des_model, converted_ckpt, strict=True)
    checkpoint['state_dict'] = converted_ckpt
    if args.des_checkpoint is None:
        checkpoint_name = args.src_checkpoint.rsplit('.', 1)[0]
        args.des_checkpoint = checkpoint_name + '.conv1d.pth'
    torch.save(checkpoint, args.des_checkpoint)
    print(f'Converted checkpoint save to {args.des_checkpoint}')


if __name__ == '__main__':
    main()
