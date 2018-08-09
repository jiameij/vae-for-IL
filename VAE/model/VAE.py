##基础版的VAE，在此基础上加东西
from common import logger, fmt_row, zipsame
import numpy as np
import tensorflow as tf
import os,sys, json, time
from collections import deque
from datetime import datetime
from model.dataset import Dset
from model.ops import optimizer_factory
from dm_control.suite import humanoid_CMU
from common.mpi_adam import MpiAdam
from common.statistics import stats

from tensorflow.python.client import timeline
import common.tf_util as U



pol_hid_size = [400, 300, 200]
pol_layers = 3
ob_shape = 63

# =========================== waveNet params============================ #
BATCH_SIZE = 1
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = '/home/jmj/VAE/model/wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False


def get_arguments():
    import argparse
    def _str_to_bool(s):
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='VAE params')
    parser.add_argument('--LSTM_size', type=int, default=300, help="bi_directional LSTM's hidden size")
    parser.add_argument('--laten_size', type=int, default=30, help="the size of laten vector Z")
    parser.add_argument('--time_steps', type=int, default=90, help="how many obs to lstm")
    parser.add_argument('--lstm_batch', type = int, default=1, help="lstm batch size")
    parser.add_argument('--state_dir_path', type = str, default="/home/jmj/08/", help="path of demonstration data dir")
    parser.add_argument('--control_timestep', type=float, default=0.02, help="env.physic.control_timestep")
    parser.add_argument('--batch_size_wave', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.') ###理论上来说应该和time_step是一样大的
    parser.add_argument('--store_metadata', type=bool, default=True,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default="./log",
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=30,
                        help='Number of global condition channels. Default: None. Expecting: Int') #########gc_channels要注意一下
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None



def load_state_dataset(data_dir_path, env, control_timestep):
    from dm_control.suite.utils import parse_amc
    converted_data = []
    files = os.listdir(data_dir_path)
    for file in files:
        if (file.endswith(".amc")):
            file_path = data_dir_path + file
            converted = parse_amc.convert(file_path, env.physics, control_timestep)  # 0.01 --> env.control_timestep()
            file_save_path = "/home/jmj/Download/subjects_08/" + file_path.split(".")[0] + ".npy"
            # np.save(file_save_path, converted.qpos)
            converted_data.append({"qpos": converted.qpos, "qvel": converted.qvel})
    state_dataset = Dset(converted_data)
    return state_dataset


def learn(env, encoder, action_decorder, state_decorder, embedding_shape,*, dataset, optimizer, logdir, batch_size, time_steps, adam_epsilon = 0.001, lr_rate = 1e-3, vae_beta = 1):
    lstm_encoder = encoder("lstm_encoder")
    ac_decoder = action_decorder("ac_decoder")
    state_decoder = state_decorder("state_decoder") #这个地方有问题
    ob = U.get_placeholder_cached(name="ob")
    obs = U.get_placeholder_cached(name="obs")  ##for encoder
    obss = U.get_placeholder_cached(name="obss")  ## for action decoder， 这个state decoder是不是也可以用, 是不是应该改成obs
    embedding = U.get_placeholder_cached(name="embedding")  ## for action decoder, 这个state decoder应该也是可以用的
    embeddingss = U.get_placeholder_cached(name="embeddingss")
    # ac = ac_decoder.pdtype.sample_placeholder([None])
    ob_next = tf.placeholder(name="ob_next", shape=[ob_shape], dtype=tf.float32)
    # obs_out = state_decoder.pdtype.sample_placeholder([None])

    # p(z) 标准正太分布
    from common.distributions import make_pdtype

    p_z_pdtype = make_pdtype(embedding_shape)
    p_z_params = U.concatenate([tf.zeros(shape=[embedding_shape], name="mean"), tf.zeros(shape=[embedding_shape], name="logstd")], axis=-1)
    p_z = p_z_pdtype.pdfromflat(p_z_params)

    # recon_loss = -tf.reduce_mean(tf.reduce_sum(state_decoder.pd.logp(obs_out), axis=0)) ###这个地方不应该是obs_out,应该是真实的值
    # recon_loss = -tf.reduce_mean(tf.reduce_sum(state_decoder.pd.logp(ob_next), axis=0)) ### 其实这里ob也是不对的,准确的来说应该是t+1时刻的ob
    recon_loss =  -state_decoder.pd.logp(ob_next)[0]
    # recon_loss = -tf.reduce_mean(tf.reduce_sum(ac_decoder.pd.logp(ac) + state_decoder.pd.logp(obs_out), axis=0)) ##这个地方还要再改
    kl_loss = lstm_encoder.pd.kl(p_z)[0] ##p(z)：标准正太分布, 这个看起来是不是也不太对！！！！
    kl_loss = tf.maximum(lstm_encoder.pd.kl(p_z)[0], tf.constant(5.00)) ##p(z)：标准正太分布, 这个看起来是不是也不太对！！！！
    vae_loss = recon_loss + vae_beta * kl_loss ###vae_loss 应该是一个batch的

    ep_stats = stats(["recon_loss", "kl_loss", "vae_loss"])
    losses = [recon_loss, kl_loss, vae_loss]
    # 均方误差去训练 action,把得到的action step 一下,得到x(t+1),然后用均方误差loss,或者可以试试交叉熵


    ## var_list
    var_list = []
    en_var_list = lstm_encoder.get_trainable_variables()
    var_list.extend(en_var_list)
    # ac_de_var_list = ac_decoder.get_trainable_variables()
    # var_list.extend(ac_de_var_list)
    state_de_var_list = state_decoder.get_trainable_variables()
    var_list.extend(state_de_var_list)
    # compute_recon_loss = U.function([ob, obs, embedding, obss, embeddingss, ac, obs_out], recon_loss)
    compute_losses = U.function([ob, obs, embedding, obss, embeddingss, ob_next], losses)
    compute_grad = U.function([ob, obs, embedding, obss, embeddingss, ob_next], U.flatgrad(vae_loss, var_list)) ###这里没有想好！！！，可能是不对的！！
    adam = MpiAdam(var_list, epsilon=adam_epsilon)


    U.initialize()
    adam.sync()

    writer = U.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    # =========================== TRAINING ===================== #
    iters_so_far = 0
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100)
    saver_encoder = tf.train.Saver(var_list = en_var_list, max_to_keep=100)
    # saver_pol = tf.train.Saver(var_list=ac_de_var_list, max_to_keep=100) ##保留一下policy的参数，但是这个好像用不到哎

    while True:
        ## 加多轮
        logger.log("********** Iteration %i ************" % iters_so_far)
        ## 要不要每一轮调整一下batch_size
        recon_loss_buffer = deque(maxlen=100)
        kl_loss_buffer = deque(maxlen=100)
        vae_loss_buffer = deque(maxlen=100)
        # i = 0
        for obs_and_next in dataset.get_next_batch(batch_size=time_steps):
            # print(i)
            # i += 1
            observations = obs_and_next[0].transpose((1, 0))
            ob_next = obs_and_next[1]
            embedding_now = lstm_encoder.get_laten_vector(observations)
            embeddings = np.array([embedding_now for _ in range(time_steps)])
            embeddings_reshape = embeddings.reshape((time_steps, -1))
            # actions = ac_decoder.act(stochastic=True, ob=observations, embedding=embeddings_reshape)
            state_outputs = state_decoder.get_outputs(observations.reshape(1, time_steps, -1), embedding_now.reshape((1, 1, -1))) ##还没有加混合高斯......乱加了一通，已经加完了
            recon_loss, kl_loss, vae_loss = compute_losses(observations, observations.reshape(batch_size,time_steps,-1),embeddings_reshape,
                              observations.reshape(1, time_steps, -1), embedding_now.reshape((1, 1, -1)), ob_next)

            g = compute_grad(observations, observations.reshape(batch_size,time_steps,-1),embeddings_reshape,
                              observations.reshape(1, time_steps, -1), embedding_now.reshape((1, 1, -1)), ob_next)
            logger.record_tabular("recon_loss", recon_loss)
            logger.record_tabular("kl_loss", kl_loss)
            logger.record_tabular("vae_loss", vae_loss)

            adam.update(g, lr_rate)
            recon_loss_buffer.append(recon_loss)
            kl_loss_buffer.append(kl_loss)
            vae_loss_buffer.append(vae_loss)
        ep_stats.add_all_summary(writer, [np.mean(recon_loss_buffer), np.mean(kl_loss_buffer),
                                          np.mean(vae_loss_buffer)], iters_so_far)
        logger.record_tabular("recon_loss", recon_loss)
        logger.record_tabular("kl_loss", kl_loss)
        logger.record_tabular("vae_loss", vae_loss)
        logger.dump_tabular()
        if(iters_so_far % 10 == 0 and iters_so_far != 0):
            save(saver=saver, sess=tf.get_default_session(), logdir=logdir, step=iters_so_far)
            save(saver=saver_encoder, sess=tf.get_default_session(),logdir="./vae_saver", step=iters_so_far)
            # save(saver=saver_pol, sess=tf.get_default_session(), logdir="pol_saver", step=iters_so_far)
        iters_so_far += 1


def train(args):
    from model.encoder import bi_direction_lstm
    from model.action_decoder import MlpPolicy
    from model.WaveNet import WaveNetModel
    U.make_session(num_cpu=1).__enter__()
    env = humanoid_CMU.stand()
    obs_space = env.physics.data.qpos
    ac_space = env.action_spec()
    def encoder(name):
        return bi_direction_lstm(name=name, obs_space=obs_space, batch_size=args.lstm_batch, time_steps= args.time_steps, LSTM_size= args.LSTM_size, laten_size = args.laten_size)
    def action_decorder(name):
        return  MlpPolicy(name=name, obs_space = obs_space, ac_space = ac_space, embedding_shape = args.laten_size, hid_size = pol_hid_size, num_hid_layers = pol_layers)
    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)
    def state_decorder(name): ##也要加个name
        return WaveNetModel(
            name = name,
            obs_shape= obs_space,
            embedding_shape= args.laten_size,
            batch_size=args.time_steps,
            dilations=wavenet_params["dilations"],
            filter_width=wavenet_params["filter_width"],
            residual_channels=wavenet_params["residual_channels"],
            dilation_channels=wavenet_params["dilation_channels"],
            skip_channels=wavenet_params["skip_channels"],
            quantization_channels=wavenet_params["quantization_channels"],
            use_biases=wavenet_params["use_biases"],
            scalar_input=wavenet_params["scalar_input"],
            initial_filter_width=wavenet_params["initial_filter_width"],
            histograms=args.histograms,
            global_condition_channels=args.gc_channels)
    state_dataset = load_state_dataset(args.state_dir_path, env, args.control_timestep)
    ##感觉数据会有点少，可以尝试多加一点走路的数
    optimizer = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum)
    learn(env=env, encoder = encoder, action_decorder=action_decorder, state_decorder=state_decorder, embedding_shape= args.laten_size ,dataset=state_dataset, optimizer = optimizer, logdir=args.logdir,
          batch_size = args.lstm_batch, time_steps = args.time_steps)

def main():
    args = get_arguments()
    train(args)

if __name__ == '__main__':
    main()