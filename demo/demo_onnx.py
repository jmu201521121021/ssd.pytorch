import  numpy as np
import onnx
import  onnxruntime
from layers import *
from data import voc, coco
import torch
import  cv2
if __name__ == '__main__':
    from data import VOC_CLASSES as labels
    num_classes = 21
    cfg = voc
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        print('prior:', priors[0, :])
        detect = Detect(num_classes, 0, 200, 0.01, 0.45)
    model = onnx.load('ssd_vgg300.onnx')
    onnx.checker.check_model(model)

    ort_session = onnxruntime.InferenceSession("ssd_vgg300.onnx")

    img = cv2.imread('../data/car.jpg')
    x = cv2.resize(img, (300, 300)).astype(np.float32)
    print(x.shape)
    print(x[0,0,:])
    x -= np.array([104.0, 117.0, 123.0])
    x = x.astype(np.float32)
    print(x.shape)
    x = np.transpose(x, (2, 0, 1))


    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    print(x.shape)
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)

    reg = torch.from_numpy(ort_outs[0][0])
    cls = torch.from_numpy(ort_outs[1][0])

    print(reg[0])
    print(cls[0][0])
    detections = detect(reg, cls, priors)


    # # scale each detection back up to the image
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            print(i-1);
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            #coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1)

            print(pt, score, label_name)
            j+=1
            cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]),(0, 0, 255), 2 )
    cv2.imshow('result', img)
    cv2.waitKey(-1)