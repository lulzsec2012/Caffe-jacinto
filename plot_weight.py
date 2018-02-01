#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : plot_weight.py
## Authors    : slwang@taurus
## Create Time: 2018-01-18:14:01:34
## Description:
## 
##
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

caffe_root = "/home/slwang/work/tools/caffe-jacinto_x"

sys.path.insert(0, caffe_root + "python")

import caffe
caffe.set_mode_cpu()


def parse_args():
    """Parse input argument"""
    parser = argparse.ArgumentParser(description="plot weight distribution")
    parser.add_argument("--weight", dest="weight", help= "set caffemodel", default=" ", type = str)
    parser.add_argument("--model", dest="model", help= "set deploy.prototxt", default=" ", type = str)
    parser.add_argument("--layer_id", dest="layer_id", help="set layer_id to plot", default=2, type = int)
    parser.add_argument("--start", dest="start", help="set start layer id", default = 0, type = int)
    parser.add_argument("--end", dest="end", help="set end layer id", default = 0, type = int)
    args = parser.parse_args()
    return args


def main(weight, model, layer_id, start, end):
   # net= caffe.Net("/home/slwang/work/tools/caffe-jacinto-models/scripts/training/imagenet_mobilenet-1.0_2018-01-02_15-58-27/test/deploy.prototxt", \
    #               "/home/slwang/imagenet_mobilenet-1.0_iter_160000.caffemodel", caffe.TEST)

    net = caffe.Net(model, weight, caffe.TEST)
    net_new = caffe.Net("/home/slwang/work/tools/caffe-jacinto-models/scripts/training/imagenet_mobilenet-1.0_2018-01-02_15-58-27/test/new_deploy.prototxt", \
                        "/home/slwang/work/models/merge_BN/merge_BN_mobilenet.caffemodel", caffe.TEST)

    param = net.params.keys();
    blob = net.blobs.keys();
    #print("blob {} \n param: {}".format(net.blobs.keys(), net.params.keys()))

    count = [[0 for i in range(100)] for j in range(len(blob))]
    ls = [[0] for i in range(100)]
    for i in range(len(blob)):
        if blob[i][0] == "c":
            data = net.params[blob[i]][0].data
            cnt = data.size
            out_channel = data.shape[0]
            in_channel = data.shape[1]
            ker_h = data.shape[2]
            ker_w = data.shape[3]
            imax = data.max()
            imin = data.min()
            ls[i] = np.linspace(imin, imax, 100)
            #print(ls)
            for n in range(out_channel):            
                for c in range(in_channel):
                    for h in range(ker_h):
                        for w in range(ker_w):                        
                            for j in range(len(ls)):
                                if data[n,c,h,w] < ls[i][j]:
                                    count[i][j] += 1
                                    break

    if start and (start != end):
        fig = plt.figure()
        num = end - start        
        for i in range(num):
            print(ls[start +  i])
            print(count[start +  i])
            plt.subplot(221+i)
            plt.title(blob[start + i])
            plt.plot(ls[start +  i], count[start + i], "r")
        plt.show()
    else:
        plt.plot(ls[layer_id], count[layer_id])
        plt.show()

if __name__ == "__main__":
    """ Parse argument """
    args = parse_args()
    weight = args.weight
    model = args.model
    layer_id = args.layer_id
    start = args.start
    end = args.end
    main(weight, model, layer_id, start, end)

                            
#for i in range (1, (len(param) - 3), 3):    
#    num = net.params[param[i]][0].data.shape[0]
#    print ("weight_shape: ", param[i], net.params[param[i]][0].data.shape[0])
#    for n in range(num):
#        net_new.params[param[i]][0].data[n, :, : ,:] = net.params[param[i]][0].data[n, :, : ,:] * net.params[param[i + 2]][0].data[n] / np.sqrt(net.params[param[i + 1]][1].data[n] + 0.00001)
#        net_new.params[param[i]][1].data[n] = net.params[param[i + 2]][1].data[n] - net.params[param[i + 2]][0].data[n] * net.params[param[i + 1]][0].data[n] / np.sqrt(net.params[param[i + 1]][1].data[n] + 0.00001)
#fc = len(param) - 1
#net_new.params[param[-1]][0].data[: , : , :, :] = net.params[param[-1]][0].data[: , : , :, :]
#net_new.params[param[-1]][1].data[:] = net.params[param[-1]][1].data[:]
 
