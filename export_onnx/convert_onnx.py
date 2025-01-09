#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: convert_onnx.py
@time: 2025/1/9 15:56
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import os
import argparse
from utils.file import MkdirSimple
from export_onnx.onnx_test import test_dir
from models import __models__
import torch

maxdisp = 192
name = "MSNet2D"


def GetArgs():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the ONNX model.")
    parser.add_argument("--height", help='Model image input height resolution', type=int, default=192)
    parser.add_argument("--width", help='Model image input height resolution', type=int, default=624)
    parser.add_argument("--test", action="store_true", help="test model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on.")
    return parser.parse_args()


class WarpModel(torch.nn.Module):
    def __init__(self):
        super(WarpModel, self).__init__()
        self.model = __models__[name](maxdisp)

    def load(self, model_path, device):
        checkpoint = torch.load(model_path)
        # self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval()

    def forward(self, image):
        width = image.shape[-1] // 2
        left_img = image[:, :, :, :width]
        right_img = image[:, :, :, width:]
        n, c, h, w = left_img.size()
        return self.model(left_img, right_img)


if __name__ == "__main__":
    args = GetArgs()
    device = torch.device(args.device)

    model_name = os.path.splitext(os.path.basename(args.model))[0].replace(" ", "_")
    output = os.path.join(args.output, model_name, f'{args.width}_{args.height}')
    onnx_file = os.path.join(output, f'SurroudnDepth_{args.width}_{args.height}_{model_name}_12.onnx')
    MkdirSimple(output)
    output_names = 'output'

    model = WarpModel()
    model.load(args.model, device)

    # Create dummy input for the model
    dummy_input = torch.randn(1, 3, args.height, args.width * 2).to(device)  # Adjust the size as needed

    # Export the depth decoder
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, onnx_file,
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True)

    if args.test:
        test_dir(onnx_file, [], output)

    print("export onnx to {}".format(onnx_file))
