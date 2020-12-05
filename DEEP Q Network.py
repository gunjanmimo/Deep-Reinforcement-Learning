import os
import numpy as np
import tensorflow as tf


class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fcl_dims=256, input_dims=(210, 160, 4), chkpt_dir="tmp/dqn"):
        self.lr = lr
        self.name = name
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, "deepqnet.ckpt")
        self.params = tf.get_collection(tf.GraphKeys)
