#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : merge_mean2bias.py
## Authors    : slwang@taurus
## Create Time: 2018-01-16:09:41:14
## Description:
## 
##
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
caffe_root = "/home/slwang/work/tools/caffe-jacinto_x"

sys.path.insert(0, caffe_root + "python")

import caffe
caffe.set_mode_cpu()

#net= caffe.Net("/home/slwang/work/tools/caffe-jacinto-models/scripts/training/imagenet_mobilenet-1.0_2018-01-02_15-58-27/test/deploy.prototxt", \
#                "/home/slwang/imagenet_mobilenet-1.0_iter_160000.caffemodel", caffe.TEST)

net_new = caffe.Net("/home/slwang/work/tools/caffe-jacinto_x/models/mergeBN/new_deploy.prototxt", \
                    "/home/slwang/work/merge_mobilenet.caffemodel", caffe.TEST)

print("blobs {}\n params {}".format(net_new.blobs.keys(), net_new.params.keys()))
param = net_new.params.keys();
blob = net_new.blobs.keys();
conv1 = net_new.params[param[1]];
out_channel = conv1[0].data.shape[0]
in_channel = conv1[0].data.shape[1]
ker_h = conv1[0].data.shape[2]
ker_w = conv1[0].data.shape[3]
mean = [103.94, 116.78, 123.68]
scale = 0.017
print("bias_shape: ", conv1[1].data.shape[0])
print("output_num {}, input_channel {}".format(out_channel, in_channel))
for n in range(out_channel):
    summ = 0.0
    for c in range(in_channel):
        for h in range(ker_h):
            for w in range(ker_w):
                conv1[0].data[n, c, h, w] = conv1[0].data[n, c, h, w] * scale
                summ += conv1[0].data[n, c, h, w] * mean[c]

    conv1[1].data[n]= conv1[1].data[n] - summ
    
net_new.save("/home/slwang/work/new_merge_mobilenet.caffemodel")    











#for i in range (1, (len(param) - 3), 3):    
#    num = net.params[param[i]][0].data.shape[0]
#    print ("weight_shape: ", param[i], net.params[param[i]][0].data.shape[0])
#    for n in range(num):
#        net_new.params[param[i]][0].data[n, :, : ,:] = net.params[param[i]][0].data[n, :, : ,:] * net.params[param[i + 2]][0].data[n] / np.sqrt(net.params[param[i + 1]][1].data[n] + 0.00001)
#        net_new.params[param[i]][1].data[n] = net.params[param[i + 2]][1].data[n] - net.params[param[i + 2]][0].data[n] * net.params[param[i + 1]][0].data[n] / np.sqrt(net.params[param[i + 1]][1].data[n] + 0.00001)
#fc = len(param) - 1
#net_new.params[param[-1]][0].data[: , : , :, :] = net.params[param[-1]][0].data[: , : , :, :]
#net_new.params[param[-1]][1].data[:] = net.params[param[-1]][1].data[:]
#print (net_new.params['conv1'][0].data)
#print (net.params['conv1'][0].data)
#print (net_new.params['conv1'][1].data)
#net_new.save("/home/slwang/work/merge_mobilenet.caffemodel")    
    





#net.save("/home/slwang/work/a_mobilenet.caffemodel")


#net1 = caffe.Net("/home/slwang/work/caffe/models/mobile/mobilenet.prototxt", \
#                "models/mobile/new_mobilenet.caffemodel", caffe.TEST)
#net1.save("models/mobile/no_bn_mobilenet.caffemodel")

