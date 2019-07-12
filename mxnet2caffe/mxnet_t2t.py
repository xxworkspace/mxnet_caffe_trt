
import os
import sys
import cv2
import time
import math
import caffe
import mxnet as mx
import numpy as np
from collections import namedtuple

prefix = "models/ensemble_L2"
epoch = 0

if __name__ == "__main__":
    time_start = time.time()
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    #print(sym)
    # print(arg_params)
    # print(aux_params)

    # get all internal outputs
    all_layers = sym.get_internals()

    sym0 = all_layers["cas_fc_128_2c_dense_output"]
    sym0 = mx.symbol.SoftmaxActivation(sym0,name="prob")

    sym1 = all_layers["cls1_fc_dense__output"]
    sym1 = mx.symbol.SoftmaxActivation(sym1,name="prob1")

    sym2 = all_layers["cls2_fc_dense__output"]
    sym2 = mx.symbol.SoftmaxActivation(sym2,name="prob2")

    sym3 = all_layers["cls3_fc_dense__output"]
    sym3 = mx.symbol.SoftmaxActivation(sym3,name="prob3")

    '''
    model0 = mx.mod.Module(symbol=sym0, label_names=None)
    model0.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    model0.set_params(arg_params, aux_params)
    model0.save_checkpoint("ensemble_L2_0",0)

    model1 = mx.mod.Module(symbol=sym1, label_names=None)
    model1.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    model1.set_params(arg_params, aux_params)
    model1.save_checkpoint("ensemble_L2_1",0)

    model2 = mx.mod.Module(symbol=sym2, label_names=None)
    model2.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    model2.set_params(arg_params, aux_params)
    model2.save_checkpoint("ensemble_L2_2",0)

    model3 = mx.mod.Module(symbol=sym3, label_names=None)
    model3.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    model3.set_params(arg_params, aux_params)
    model3.save_checkpoint("ensemble_L2_3",0)
    '''
    sym  = mx.symbol.concat(sym0,sym1,sym2,sym3,dim = 0,name = "prob_concat")

	#sym.save()
    #sym.save("ensemble_L2-symbol.json")
    # rebuild
    model = mx.mod.Module(symbol=sym, label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    model.set_params(arg_params, aux_params)

    model.save_checkpoint("ensemble_L2_XX",0)
