import numpy as np
import tensorflow as tf

import argparse
import time
import os
import pickle

from utils import DataLoader, parameter
from model import Model


def train(args):
    if tf.__version__.startswith('1.2'):
        data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale, data_dir ='/home/easton/Data')
    elif tf.__version__.startswith('1.1'):
        data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale, data_dir ='../../data')

    if args.model_dir != '' and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = Model(args)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'log'), sess.graph)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            v_x, v_y = data_loader.validation_data()
            valid_feed = {model.input_data: v_x, model.target_data: v_y, model.state_in: sess.run(model.state_in)}
            state = sess.run(model.state_in)

            for b in range(data_loader.num_batches):
                i = e * data_loader.num_batches + b
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.target_data: y, model.state_in: state}
                train_loss_summary, train_loss, state, _ = sess.run([model.train_loss_summary, model.cost, model.state_in, model.train_op], feed)
                summary_writer.add_summary(train_loss_summary, i)

                valid_loss_summary, valid_loss, = sess.run([model.valid_loss_summary, model.cost], valid_feed)
                summary_writer.add_summary(valid_loss_summary, i)

                end = time.time()
                print(
                    "{}/{} (epoch {}), train_loss = {:.5f}, valid_loss = {:.5f}, time/batch = {:.1f}"  \
                    .format(
                        i,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        train_loss, valid_loss, end - start))
                if i % args.save_every == 0:
                    checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = i)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    with open('parameter.txt', 'r') as f:
        parser =parameter()
        for line in f:
            if 'for model.py' in line: break
        parser.rnn_size         =int(f.readline().split()[0])
        parser.num_layers  =int(f.readline().split()[0])
        parser.batch_size         =int(f.readline().split()[0])
        parser.seq_length       =int(f.readline().split()[0])
        parser.num_epochs       =int(f.readline().split()[0])
        parser.save_every            =int(f.readline().split()[0])
        parser.model_dir =f.readline().split()[0]
        parser.grad_clip       =float(f.readline().split()[0])
        parser.learning_rate            =float(f.readline().split()[0])
        parser.decay_rate       =float(f.readline().split()[0])
        parser.num_mixture            =int(f.readline().split()[0])
        parser.data_scale       =float(f.readline().split()[0])
        parser.keep_prob            =float(f.readline().split()[0])

        train(parser)
