from __future__ import print_function, division

from models import __models__
from utils import *

maxdisp = 192
name = "MSNet2D"

model = __models__[name](maxdisp)
model.cuda()
model.eval()

def test():
    print("Generating the disparity maps...")

    width = 624
    height = 192
    dummy_input_L = torch.randn(1, 3, height, width, device='cuda:0')
    dummy_input_R = torch.randn(1, 3, height, width, device='cuda:0')
    input_names = ['L', 'R']
    output_names = ['output']
    torch.onnx.export(
        model,
        (dummy_input_L,dummy_input_R),
        "./{}.onnx".format(name),
        verbose=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names)

if __name__ == '__main__':
    test()
