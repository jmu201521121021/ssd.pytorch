import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import numpy as np


import onnx
import  onnxruntime
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd, build_ssd_onnx

def converToOnxx(torch_model):
    torch_model.eval()
    batch_size = 1
    x = torch.randn(batch_size, 3, 300, 300, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "ssd_vgg300.onnx",  # where to save the model (can be a file or file-like object)
                      verbose = True,
                  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['reg', 'cls'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                      #               'reg': {0: 'batch_size'},
                      #               'cls': {0 :'batch_size'}}
                    )
    import onnx

    onnx_model = onnx.load("ssd_vgg300.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("ssd_vgg300.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)

if __name__ == '__main__':

    net = build_ssd_onnx('test', 300, 21)
    net.load_weights('../weights/ssd300_mAP_74.9.pth')
    converToOnxx(net)
    print('convert pytorch to onnx model successfully!!!\n')

