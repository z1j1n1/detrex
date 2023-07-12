import torch
import os

cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
dets = torch.load("/data/jiangning/detrex/output_2012test/VOC_2012_test_output.pth")
for idx, cls_name in enumerate(cls):
    fp = open(f'/data/jiangning/detrex/output_2012test/dets/comp3_det_test_{cls_name}.txt','w')
    for line in dets[idx]:
        fp.write(line+'\r')
    fp.close()
