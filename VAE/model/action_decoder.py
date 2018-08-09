import numpy as np
import common.tf_util as U
import tensorflow as tf
from common.distributions import make_pdtype
from common.mpi_running_mean_std import RunningMeanStd

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, obs_space, ac_space, embedding_shape, hid_size, num_hid_layers, gaussian_fixed_var=True):
        self.pdtype = pdtype = make_pdtype(ac_space.shape[0])
        batch_size = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[batch_size, obs_space.shape[0]])
        embedding = U.get_placeholder(name="embedding", dtype=tf.float32, shape=[batch_size, embedding_shape]) ##这里我觉得是一个embedding 的值扩展成sequence_len大小，暂时先不管，等具体做到这里的时候再处理

        # 正则化一下
        last_out = U.concatenate([ob, embedding], axis=1)
        with tf.variable_scope("ac_de_filter"):
            self.ac_rms = RunningMeanStd(shape=obs_space.shape[0] + embedding_shape)

        last_out = tf.clip_by_value((last_out - self.ac_rms.mean) / self.ac_rms.std, -5.0, 5.0)

        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size[i], "ac_de%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space.shape[0], int):
            self.mean = U.dense(last_out, pdtype.param_shape()[0]//2, "ac_de_final", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([self.mean, self.mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "ac_de_final", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob, embedding], ac)
        self._get_pol_mean = U.function([ob, embedding], self.mean)

    def act(self, stochastic, ob, embedding):
        # embeddings = np.array([embedding[0] for _ in range(batch_size)])
        ac1 =  self._act(stochastic, ob, embedding)
        return ac1
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

    def get_policy_mean(self, ob, embedding):
        pol_mean = self._get_pol_mean(ob, embedding)
        return pol_mean[0] ##返回policy参数的均值