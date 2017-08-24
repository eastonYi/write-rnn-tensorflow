import tensorflow as tf

import numpy as np
import random
import copy
import pdb

class Model():
  def __init__(self, args, infer=False):
    self.args = args
    if infer:
      args.batch_size = 1
      args.seq_length = 1

    if tf.__version__.startswith('1.2'):
        cell =tf.nn.rnn_cell.LSTMCell

        stacked_rnn =[tf.contrib.rnn.DropoutWrapper(cell(args.rnn_size), output_keep_prob = args.keep_prob) if (infer == False and args.keep_prob < 1) else cell(args.rnn_size) \
                      for _ in range(args.num_layers)]
        cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn)

    elif tf.__version__.startswith('1.1'):
        cell =tf.contrib.rnn.LSTMCell

        stacked_rnn =[tf.contrib.rnn.DropoutWrapper(cell(args.rnn_size), output_keep_prob = args.keep_prob) if (infer == False and args.keep_prob < 1) else cell(args.rnn_size) \
                      for _ in range(args.num_layers)]
        cell_fw = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    self.cell = cell_fw

    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, 3], name='data_in')
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, 3], name='targets')
    zero_state = cell_fw.zero_state(batch_size=args.batch_size, dtype=tf.float32)
    self.state_in =tuple(tuple(x) for x in zero_state)

    self.num_mixture = args.num_mixture
    NOUT = 1 + self.num_mixture * 6 # end_of_stroke + num_mixture*(pi+mu_x+mu_y+sigma_x+sigma_y+rho)

    inputs = tf.unstack(self.input_data, axis=1)

    with tf.variable_scope('rnnlm'):
      output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    '''
    inputs: A length T list of inputs, each a `Tensor` of shape
      `[batch_size, input_size]`, or a nested tuple of such elements.
    '''
    outputs, state_out =tf.contrib.rnn.static_rnn(
        cell =self.cell,
        inputs =inputs,
        initial_state =self.state_in,
        scope='rnnlm')

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, args.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)

    self.state_out =tuple(tuple(x) for x in state_out)

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_data,[-1, 3])
    [x1_data, x2_data, eos_data] = tf.split(axis=1, num_or_size_splits=3, value=flat_target_data)

    def get_mixture_coef(output):
      # returns the tf slices containing mdn dist params
      # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
    #   z = tf.reshape(output, [-1, self.seq_length, NOUT])
      z_eos = output[:, 0:1]
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(axis=1, num_or_size_splits=6, value=output[:, 1:])

      # process output z's into MDN paramters

      # end of stroke signal
      z_eos = tf.sigmoid(z_eos) # should be negated, but doesn't matter.
      # softmax all the pi's:
      z_pi =tf.nn.softmax(z_pi, 1)
      z_sigma1 = tf.exp(z_sigma1)
      z_sigma2 = tf.exp(z_sigma2)
      z_corr = tf.tanh(z_corr)

      return [z_eos, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]

    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
        ## you can view the x_1 , x_2, ... is the scalar, but actually they are all 2D(batch*time, num_gaussian) tensor
      # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
      norm1 = tf.subtract(x1, mu1)
      norm2 = tf.subtract(x2, mu2)
      s1s2 = tf.multiply(s1, s2)
      z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
      negRho = 1-tf.square(rho)
      result = tf.exp(tf.div(-z,2*negRho))
      denom = 2*np.pi*tf.multiply(s1s2, tf.sqrt(negRho))
      result = tf.div(result, denom)
      return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, x1_data, x2_data, eos_data):
      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
      # implementing eq # 26 of http://arxiv.org/abs/1308.0850
      epsilon = 1e-20
      result1 = tf.multiply(result0, z_pi)
      result1 = tf.reduce_sum(result1, 1, keep_dims=True)
      result1 = -tf.log(tf.maximum(result1, 1e-20)) # at the beginning,
      result2 = tf.multiply(z_eos, eos_data) + tf.multiply(1-z_eos, 1-eos_data)
      result2 = -tf.log(result2)
      result = result1 + result2

      return tf.reduce_sum(result)

    # below is where we need to do MDN splitting of distribution params

    o_eos, o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr= get_mixture_coef(output)

    self.eos = o_eos
    self.pi = o_pi
    self.mu1 = o_mu1
    self.mu2 = o_mu2
    self.sigma1 = o_sigma1
    self.sigma2 = o_sigma2
    self.corr = o_corr

    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, x1_data, x2_data, eos_data)
    self.cost = lossfunc / (args.batch_size * args.seq_length)

    self.train_loss_summary = tf.summary.scalar('train_loss', self.cost)
    self.valid_loss_summary = tf.summary.scalar('validation_loss', self.cost)

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

  def sample(self, sess, num=1200):
    def get_pi_idx(x, pdf):
      N = pdf.size
      accumulate = 0
      for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
          return i
      print('error with sampling ensemble')
      return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
      mean = [mu1, mu2]
      cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
      x = np.random.multivariate_normal(mean, cov, 1)
      return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 3), dtype=np.float32)
    prev_x[0, 0, 2] = 1 # initially, we want to see beginning of new stroke
    prev_state =sess.run(self.state_in)

    strokes = np.zeros((num, 3), dtype=np.float32)
    mixture_params = []

    for i in range(num):

      feed = {self.input_data: prev_x, self.state_in:prev_state}

    #   pdb.set_trace()
      o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, next_state = \
      sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.eos, self.state_out], feed)

      # as the batch_size * seq_length =1 , o_pi.shape =[1 * num_gaussian]
      idx = get_pi_idx(random.random(), o_pi[0])

      eos = 1 if random.random() < o_eos[0][0] else 0

      next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

      strokes[i,:] = [next_x1, next_x2, eos]

      params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_eos[0]]
      mixture_params.append(params)

      prev_x[0][0] = np.array([next_x1, next_x2, eos], dtype=np.float32)
      prev_state =next_state

    strokes[:,0:2] *= self.args.data_scale
    return strokes, mixture_params
