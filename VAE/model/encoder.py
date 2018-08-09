import tensorflow as tf
import common.tf_util as U
from tensorflow.contrib import rnn
from common.distributions import make_pdtype
from common.mpi_running_mean_std import RunningMeanStd

class bi_direction_lstm(object):
    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, obs_space, batch_size, time_steps, LSTM_size, laten_size, gaussian_fixed_var=True): ##等会儿要重点看一下var有没有更新
        self.pdtype = pdtype = make_pdtype(laten_size)
        obs = U.get_placeholder("obs", dtype=tf.float32, shape = [batch_size, time_steps, obs_space.shape[0]])
        # 正则化
        with tf.variable_scope("obfilter"): ## 看看有没有起效果，我觉得是其效果考虑的
            self.obs_rms = RunningMeanStd(shape=obs_space.shape)

        obz = tf.clip_by_value((obs - self.obs_rms.mean) / self.obs_rms.std, -5.0, 5.0)

        lstm_fw_cell = rnn.BasicLSTMCell(LSTM_size, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(LSTM_size, forget_bias=1.0)
        outputs, output_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, obz, dtype=tf.float32)
        outputs_average = tf.reduce_mean(outputs[0], axis=1)
        if gaussian_fixed_var and isinstance(laten_size, int):
            self.mean = U.dense(outputs_average, pdtype.param_shape()[0] // 2, "dblstmfin", U.normc_initializer(0.01))
            self.logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                     initializer=tf.constant_initializer(0.1))
            pdparam = U.concatenate([self.mean, self.mean * 0.0 + self.logstd], axis=1)

        else:
            pdparam = U.dense(outputs_average, pdtype.param_shape()[0], "dblstmfin", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)
        self._encode = U.function([obs], self.pd.sample())
        self._get_mean = U.function([obs], self.mean)

    def get_laten_vector(self, obs):
        embedding = self._encode(obs[None])
        # mean = self._get_mean(obs[None])
        return embedding

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)




    ################# 后面可能会设计到kl散度之类的问题吧