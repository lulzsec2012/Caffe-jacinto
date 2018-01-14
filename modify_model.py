#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : modify_model.py
## Authors    : slwang@aries
## Create Time: 2017-12-13:15:55:03
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

net= caffe.Net("/home/slwang/work/tools/caffe-jacinto-models/scripts/training/imagenet_mobilenet-1.0_2018-01-02_15-58-27/test/deploy.prototxt", \
                "/home/slwang/imagenet_mobilenet-1.0_iter_160000.caffemodel", caffe.TEST)

net_new = caffe.Net("/home/slwang/work/tools/caffe-jacinto-models/scripts/training/imagenet_mobilenet-1.0_2018-01-02_15-58-27/test/new_deploy.prototxt", \
                    "/home/slwang/work/_iter_1.caffemodel", caffe.TEST)
#net.save("models/mobile/new_mobilenet.caffemodel")
#print("blobs {}\n params {}".format(net.blobs.keys(), net.params.keys()))
#print("****************************************************************")
#print("blobs {}\n params {}".format(net_new.blobs.keys(), net_new.params.keys()))
#print(net.params['conv1/bn'][2].data)
param = net.params.keys();
blob = net.blobs.keys();
for i in range (1, (len(param) - 3), 3):    
    num = net.params[param[i]][0].data.shape[0]
    print ("weight_shape: ", param[i], net.params[param[i]][0].data.shape[0])
    for n in range(num):
        net_new.params[param[i]][0].data[n, :, : ,:] = net.params[param[i]][0].data[n, :, : ,:] * net.params[param[i + 2]][0].data[n] / np.sqrt(net.params[param[i + 1]][1].data[n] + 0.00001)
        net_new.params[param[i]][1].data[n] = net.params[param[i + 2]][1].data[n] - net.params[param[i + 2]][0].data[n] * net.params[param[i + 1]][0].data[n] / np.sqrt(net.params[param[i + 1]][1].data[n] + 0.00001)
fc = len(param) - 1
net_new.params[param[-1]][0].data[: , : , :, :] = net.params[param[-1]][0].data[: , : , :, :]
net_new.params[param[-1]][1].data[:] = net.params[param[-1]][1].data[:]
print (net_new.params['conv1'][0].data)
print (net.params['conv1'][0].data)
print (net_new.params['conv1'][1].data)
net_new.save("/home/slwang/work/merge_mobilenet.caffemodel")    
#conv = []
#bn = []
#scale = []
#for i in range(1, (len(param) - 3), 3):
#    conv.append(param[i])
#    bn.append(param[i+1])
#    scale.append(param[i+2])
#for i in range(len(conv)):
#    num = net.params[conv[i]][0].data.shape[0]
#    print ("weight_shape: ", net.params[conv[i]][0].data.shape[0], conv[i])
#    for n in range(num):
#        net.params[conv[i]][0].data[n, :, :, :] = net.params[conv[i]][0].data[n, :, :, :] * net.params[scale[i]][0].data[n] / np.sqrt(net.params[bn[i]][1].data[n] + 0.00001)
#        net.params[scale[i]][1].data[n] = net.params[scale[i]][1].data[n] - net.params[scale[i]][0].data[n] * net.params[bn[i]][0].data[n] / np.sqrt(net.params[bn[i]][1].data[n] + 0.00001)
#        net.params[scale[i]][0].data[n] = 1 
#        net.params[bn[i]][0].data[n] = 0
#        net.params[bn[i]][1].data[n] = 1
#print (net.params['conv1'][0].data)

#param_old = net.params.keys()
#param_new = net.params.keys()
#for i in len(param):
    





#net.save("/home/slwang/work/a_mobilenet.caffemodel")


#net1 = caffe.Net("/home/slwang/work/caffe/models/mobile/mobilenet.prototxt", \
#                "models/mobile/new_mobilenet.caffemodel", caffe.TEST)
#net1.save("models/mobile/no_bn_mobilenet.caffemodel")
