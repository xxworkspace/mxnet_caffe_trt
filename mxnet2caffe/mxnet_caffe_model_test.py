
import os
import sys
import time
import math
import mxnet as mx
import caffe
from caffe import NetSpec
from caffe.proto import caffe_pb2

import cv2
import argparse
import numpy as np
from collections import namedtuple

parser = argparse.ArgumentParser(description='Test Caffe model Result!')

parser.add_argument('--caffe',required=True,type=str,help='path to caffe model name')
parser.add_argument('--caffe_output',type=str,default="prob",help='blob name of caffe model output')

parser.add_argument('--mxnet',required=True,type=str,help='path to mxnet model name')
parser.add_argument('--mxnet_epoch',type=int,default=0,help='epoch of mxnet model')
parser.add_argument('--image_path' ,required=True,type=str,help='path to test image')

parser.add_argument('--mean',type=str,default="0,0,0", help='mean used to process image,like "128,128,128"')
parser.add_argument('--scale',type=str,default="1,1,1",help='scale used to process image, like "1.0,1.0,1.0"')

args = parser.parse_args()
means= [float(x) for x in args.mean.split(",")]
scale= [float(x) for x in args.scale.split(",")]

def single_input(path):
    img = cv2.imread(path)
	#load
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    #hwc -> chw
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img.astype("float32")

    img = [scale[0]*(img[0] - means[0]),scale[1]*(img[1] - means[1]),scale[2]*(img[2] - means[2])]
    img = np.array(img)
    img = img[np.newaxis, :]
    return img

def caffe_test():
    deploy      = args.caffe + '.prototxt'
    caffe_model = args.mxnet + '.caffemodel'

    net = caffe.Net(deploy,caffe_model,caffe.TEST)
    img = single_input(args.image_path)
    net.blobs['data'].data[...] = img

    out = net.forward()
    prob= net.blobs[args.caffe_output].data
    print prob

caffe_test()
if __name__ == "__main__":
    time_start = time.time()
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.mxnet, args.mxnet_epoch)
    #print(sym)
    # print(arg_params)
    # print(aux_params)

    # get all internal outputs
    #all_layers = sym.get_internals()
    #print all_layers

    model = mx.mod.Module(symbol=sym, label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    model.set_params(arg_params, aux_params)
    Batch = namedtuple("batch", ['data'])

    img = single_input(args.image_path)
    array = mx.nd.array(img)
    model.forward(Batch([array]))
    vector2 = model.get_outputs()[0].asnumpy()
    vector2 = np.squeeze(vector2)
    print vector2
