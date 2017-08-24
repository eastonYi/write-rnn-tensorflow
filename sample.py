import numpy as np
import tensorflow as tf

import time
import os
import pickle

from utils import *
from model import Model
import random

import svgwrite
from IPython.display import SVG, display



# main code (not in a main function since I want to run this script in IPython as well).
with open('parameter.txt', 'r') as f:
    sample_args =parameter()
    for line in f:
        if 'for sample.py' in line: break
    sample_args.filename         =f.readline().split()[0]
    sample_args.sample_length  =int(f.readline().split()[0])
    sample_args.scale_factor  =int(f.readline().split()[0])
    sample_args.model_dir         =f.readline().split()[0]
    if f.readline().split()[0] =='True':
        sample_args.freeze_graph =True
    else:
        sample_args.freeze_graph =False

with open(os.path.join(sample_args.model_dir, 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

model = Model(saved_args, True)
sess = tf.InteractiveSession()
#saver = tf.train.Saver(tf.all_variables())
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(sample_args.model_dir)
print("loading model: ", ckpt.model_checkpoint_path)

saver.restore(sess, ckpt.model_checkpoint_path)

def sample_stroke():
  [strokes, params] = model.sample(sess, sample_args.sample_length)
  draw_strokes(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.normal.svg')
  draw_strokes_random_color(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.color.svg')
  draw_strokes_random_color(strokes, factor=sample_args.scale_factor, per_stroke_mode = False, svg_filename = sample_args.filename+'.multi_color.svg')
  draw_strokes_eos_weighted(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.eos_pdf.svg')
  draw_strokes_pdf(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.pdf.svg')
  return [strokes, params]


def freeze_and_save_graph(sess, folder, out_nodes, as_text=False):
    ## save graph definition
    graph_raw = sess.graph_def
    graph_frz = tf.graph_util.convert_variables_to_constants(sess, graph_raw, out_nodes)
    ext = '.txt' if as_text else '.pb'
    #tf.train.write_graph(graph_raw, folder, 'graph_raw'+ext, as_text=as_text)
    tf.train.write_graph(graph_frz, folder, 'graph_frz'+ext, as_text=as_text)


# if(sample_args.freeze_graph):
#     freeze_and_save_graph(sess, sample_args.model_dir, ['data_out_mdn', 'data_out_eos', 'state_out'], False)

[strokes, params] = sample_stroke()
with open('svg/strokes'+sample_args.filename[-1]+'.pkl', 'wb') as f:
    saved_args = pickle.dump([strokes, params], f)
