
import common.tf_util as U
import tensorflow as tf
from common.distributions import make_pdtype
from common.mpi_running_mean_std import RunningMeanStd


class MlpPolicy_state(object):
    recurrent = False
    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, obs_space, embedding_shape, hid_size, num_hid_layers, gaussian_fixed_var=True):
        self.pdtype = pdtype = make_pdtype(obs_space.shape[0])
        batch_size = None

        ob_input = U.get_placeholder(name="ob", dtype=tf.float32, shape=[batch_size, obs_space.shape[0]])
        embedding = U.get_placeholder(name="embedding", dtype=tf.float32, shape=[batch_size, embedding_shape]) ##这里我觉得是一个embedding 的值扩展成sequence_len大小，暂时先不管，等具体做到这里的时候再处理

        last_out = U.concatenate([ob_input, embedding], axis=1) ##这里只有policy, 没有 value function, 还有这个要看看concatenate的对不对
        # 正则化
        with tf.variable_scope("state_de_filter"):
            self.state_rms = RunningMeanStd(shape=obs_space.shape[0] + embedding_shape)

        input_z = tf.clip_by_value((last_out - self.state_rms.mean) / self.state_rms.std, -5.0, 5.0)


        for i in range(num_hid_layers):
            input_z = tf.nn.tanh(U.dense(input_z, hid_size[i], "state_de%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(obs_space.shape[0], int):
            mean = U.dense(input_z, pdtype.param_shape()[0]//2, "state_de_final", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "state_de_final", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        self._act = U.function([ob_input, embedding], self.pd.sample())

    def get_outputs(self, ob, embedding):
        ac1 =  self._act(ob, embedding)
        return ac1

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
