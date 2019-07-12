# -*- coding:utf-8 -*-

import sys
import json
import argparse
from prototxt_basic import *
import logging
logging.basicConfig(level = logging.INFO)

parser = argparse.ArgumentParser(description='Convert MXNet jason to Caffe prototxt')
parser.add_argument('--mx-json',required=True,type=str, help='path to mxnet model json')
parser.add_argument('--cf-prototxt',required=True,type=str,help='path to save caffe model prototxt')

args = parser.parse_args()

with open(args.mx_json) as json_file:
  jdata = json.load(json_file)

with open(args.cf_prototxt, "w") as prototxt_file:
  for i_node in range(0,len(jdata['nodes'])):

    node_i    = jdata['nodes'][i_node]

    if str(node_i['op']) == 'null' and str(node_i['name']) != 'data':
      continue
    '''
    print('{}, \top:{}, name:{} -> {}'.format(i_node,node_i['op'].ljust(20),
                                        node_i['name'].ljust(30),
                                        node_i['name']).ljust(20))
                                  '''
    ##node[i] op  name  param  input
    info = node_i
    
    info['top'] = info['name']
    if "_fwd" in info['name']:
      info['top'] = node_i['name'].replace('_fwd','')
    info['bottom'] = []
    info['params'] = []

    #print(node_i["name"])
    #print(info['top'],node_i['op'])
    for input_idx_i in node_i['inputs']:
      #print input_idx_i
      # jdata['nodes'][input_idx_i[0]]  jdana['nodes'][input_index]
      input_i = jdata['nodes'][input_idx_i[0]]
      #print(input_i["name"])
      # 找 bottom
      if str(input_i['op']) != 'null' or (str(input_i['name']) == 'data'):
        if "_fwd" in str(input_i['name']):#mxnet gluon model
          info['bottom'].append(input_i['name'].replace('_fwd',''))
        else:
          info['bottom'].append(str(input_i['name']))
      #
      if str(input_i['op']) == 'null':
        info['params'].append(str(input_i['name']))
        #if not str(input_i['name']).startswith(str(node_i['name'])):
        #  logging.info('           use shared weight -> %s'% str(input_i['name']))
        #  info['share'] = True
      
    write_node(prototxt_file, info)

