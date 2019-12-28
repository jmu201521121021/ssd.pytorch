import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2

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
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['reg', 'cls'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'reg': {0: 'batch_size'},
                                    'cls': {0 :'batch_size'}})
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
    from data import VOC_CLASSES as labels

    net = build_ssd('test', 300, 21)  # initialize SSD
    net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')
    #
    # net = build_ssd_onnx('test', 300, 21)
    # converToOnxx(net)

    top_k = 10
    img = cv2.imread('../data/example.jpg')
    x = cv2.resize(img, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    # scale each detection back up to the image
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)
    detections = y.data
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            #coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1)

            print(pt, score, label_name)
            j+=1
            cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]),(0, 0, 255), 2 )
    cv2.imshow('result', img)
    cv2.waitKey(-1)