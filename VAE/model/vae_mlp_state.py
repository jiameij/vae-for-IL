##基础版的VAE，在此基础上加东西
from common import logger
import numpy as np
import tensorflow as tf
import os,sys
from datetime import datetime
from model.dataset import Dset
from collections import deque
from dm_control.suite import humanoid_CMU
from common.mpi_adam import MpiAdam
from common.statistics import stats

import common.tf_util as U

# BATCH_SIZE = 1
#
# LOGDIR_ROOT = './logdir'
# NUM_STEPS = int(1e5)
# LEARNING_RATE = 1e-3
# WAVENET_PARAMS = '/home/jmj/VAE/model/wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# SAMPLE_SIZE = 100000
# L2_REGULARIZATION_STRENGTH = 0
# SILENCE_THRESHOLD = 0.3
# EPSILON = 0.001
# MOMENTUM = 0.9
# MAX_TO_KEEP = 5
# METADATA = False


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='VAE params')
    # for bi-directional lstm
    parser.add_argument('--LSTM_size', type=int, default=300, help="bi_directional LSTM's hidden size")
    parser.add_argument('--laten_size', type=int, default=30, help="the size of laten vector Z")
    parser.add_argument('--time_steps', type=int, default=32, help="how many obs to lstm")
    parser.add_argument('--lstm_batch', type = int, default=1, help="lstm batch size")

    # for policy network
    parser.add_argument('--pol_hid_size', type = list, default=[400, 300, 200])
    parser.add_argument('--pol_layers', type = int, default=3)
    # parser.add_argument('--batch_size', type = int, default=)

    # for state decoder
    parser.add_argument('--state_de_hid_size', type = list, default= [200, 100])
    parser.add_argument('--state_de_hid_num', type = int, default= 2)

    # for load demonstration dataset
    parser.add_argument('--state_dir_path', type = str, default="/home/jmj/08/", help="path of demonstration data dir")
    parser.add_argument('--control_timestep', type=float, default=0.01, help="env.physic.control_timestep")

    # for train vae
    parser.add_argument('--lr_rate', type = float, default= 1e-3)
    parser.add_argument('--epsilon', type = float, default=0.001)
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
    parser.add_argument('--checkpoint_every', type=int, default=100)

    parser.add_argument('--max_checkpoints', type=int, default=5,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(5) + '.')
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
            converted_data.append({"qpos": converted.qpos, "qvel": converted.qvel})
    state_dataset = Dset(converted_data)
    return state_dataset


def learn(encoder, action_decorder, state_decorder, embedding_shape,*, dataset, logdir, batch_size, time_steps, epsilon = 0.001, lr_rate = 1e-3):
    lstm_encoder = encoder("lstm_encoder")
    ac_decoder = action_decorder("ac_decoder")
    state_decoder = state_decorder("state_decoder") #换成了mlp
    obs = U.get_placeholder_cached(name="obs")  ##for encoder

    ob = U.get_placeholder_cached(name="ob")
    embedding = U.get_placeholder_cached(name="embedding")

    # obss = U.get_placeholder_cached(name="obss")  ## for action decoder， 这个state decoder是不是也可以用, 是不是应该改成obs
    #   ## for action decoder, 这个state decoder应该也是可以用的
    # embeddingss = U.get_placeholder_cached(name="embeddingss")
    ac = ac_decoder.pdtype.sample_placeholder([None])
    obs_out = state_decoder.pdtype.sample_placeholder([None])

    # p(z) 标准正太分布, state先验分布？？？是不是应该换成demonstration的标准正态分布？？？？ 可以考虑一下这个问题
    from common.distributions import make_pdtype

    p_z_pdtype = make_pdtype(embedding_shape)
    p_z_params = U.concatenate([tf.zeros(shape=[embedding_shape], name="mean"), tf.zeros(shape=[embedding_shape], name="logstd")], axis=-1)
    p_z = p_z_pdtype.pdfromflat(p_z_params)

    recon_loss = -tf.reduce_mean(tf.reduce_sum(ac_decoder.pd.logp(ac) + state_decoder.pd.logp(obs_out), axis=0)) ##这个地方还要再改
    kl_loss = lstm_encoder.pd.kl(p_z) ##p(z)：标准正太分布, 这个看起来是不是也不太对！！！！
    vae_loss = recon_loss + kl_loss ###vae_loss 应该是一个batch的

    ep_stats = stats(["recon_loss", "kl_loss", "vae_loss"])
    losses = [recon_loss, kl_loss, vae_loss]

    ## var_list
    var_list = []
    en_var_list = lstm_encoder.get_trainable_variables()
    var_list.extend(en_var_list)
    # ac_de_var_list = ac_decoder.get_trainable_variables()
    # var_list.extend(ac_de_var_list)
    state_de_var_list = state_decoder.get_trainable_variables()
    var_list.extend(state_de_var_list)
    # compute_recon_loss = U.function([ob, obs, embedding, obss, embeddingss, ac, obs_out], recon_loss)
    compute_losses = U.function([obs, ob, embedding, ac, obs_out], losses)
    compute_grad = U.function([obs, ob, embedding, ac, obs_out], U.flatgrad(vae_loss, var_list)) ###这里没有想好！！！，可能是不对的！！
    adam = MpiAdam(var_list, epsilon=epsilon)


    U.initialize()
    adam.sync()

    writer = U.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    # =========================== TRAINING ===================== #
    iters_so_far = 0
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=100)
    saver_encoder = tf.train.Saver(var_list = en_var_list, max_to_keep=100)
    # saver_pol = tf.train.Saver(var_list=ac_de_var_list, max_to_keep=100) ##保留一下policy的参数，但是这个好像用不到哎

    while True:
        logger.log("********** Iteration %i ************" % iters_so_far)

        recon_loss_buffer = deque(maxlen=100)
        kl_loss_buffer = deque(maxlen=100)
        vae_loss_buffer = deque(maxlen=100)

        for observations in dataset.get_next_batch(batch_size=time_steps):
            observations = observations.transpose((1, 0))
            embedding_now = lstm_encoder.get_laten_vector(observations)
            embeddings = np.array([embedding_now for _ in range(time_steps)])
            embeddings_reshape = embeddings.reshape((time_steps, -1))
            actions = ac_decoder.act(stochastic=True, ob=observations, embedding=embeddings_reshape)
            state_outputs = state_decoder.get_outputs(observations.reshape(time_steps, -1, 1),
                                                      embeddings)  ##还没有加混合高斯......乱加了一通，已经加完了
            recon_loss, kl_loss, vae_loss = compute_losses(observations,
                                                           observations.reshape(batch_size, time_steps, -1),
                                                           embeddings_reshape,
                                                           observations.reshape(time_steps, -1, 1), embeddings, actions,
                                                           state_outputs)

            g = compute_grad(observations, observations.reshape(batch_size, time_steps, -1), embeddings_reshape,
                             observations.reshape(time_steps, -1, 1), embeddings, actions, state_outputs)
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
        if (iters_so_far % 10 == 0 and iters_so_far != 0):
            save(saver=saver, sess=tf.get_default_session(), logdir=logdir, step=iters_so_far)
            save(saver=saver_encoder, sess=tf.get_default_session(), logdir="./vae_saver", step=iters_so_far)
            # save(saver=saver_pol, sess=tf.get_default_session(), logdir="pol_saver", step=iters_so_far)
        iters_so_far += 1

        # observations = dataset.get_next_batch(batch_size=time_steps)[0].transpose((1, 0))
        #
        # embedding = lstm_encoder.get_laten_vector(observations)
        # embeddings = np.array([embedding[0] for _ in range(time_steps)])
        # actions = ac_decoder.act(stochastic=True, ob=observations, embedding=embeddings)
        # # state_outputs = []
        # # for i in range(time_steps):
        # #     output = state_decoder.get_outputs(observations[i].reshape(1, -1), embedding)
        # #     state_outputs.append(output)
        # # state_outputs = np.array(state_outputs)
        # state_outputs = state_decoder.get_outputs(observations, embeddings) ###这个地方有问题
        #
        # recon_loss, kl_loss, vae_loss = compute_losses(observations.reshape((1,time_steps, -1)), observations,embeddings, actions, state_outputs)
        #
        # g = compute_grad(observations.reshape((1,time_steps, -1)), observations ,embeddings, actions, state_outputs)
        # adam.update(g, lr_rate)
        # ep_stats.add_all_summary(writer, [np.mean(recon_loss), np.mean(kl_loss),
        #                                   np.mean(vae_loss)], iters_so_far)
        # logger.record_tabular("recon_loss", recon_loss)
        # logger.record_tabular("kl_loss", kl_loss)
        # logger.record_tabular("vae_loss", vae_loss)
        # logger.dump_tabular()
        # if(iters_so_far % 50 == 0 and iters_so_far != 0):
        #     save(saver=saver, sess=tf.get_default_session(), logdir=logdir, step=iters_so_far)
        #     # save(saver=saver_encoder, sess=tf.get_default_session(),logdir="./vae_saver", step=iters_so_far)
        #     # save(saver=saver_pol, sess=tf.get_default_session(), logdir="pol_saver", step=iters_so_far)
        # iters_so_far += 1


def train(args):
    from model.encoder import bi_direction_lstm
    from model.action_decoder import MlpPolicy
    from model.mlp_state_decoder import MlpPolicy_state
    U.make_session(num_cpu=1).__enter__()
    env = humanoid_CMU.stand()
    obs_space = env.physics.data.qpos
    ac_space = env.action_spec()
    def encoder(name):
        return bi_direction_lstm(name=name, obs_space=obs_space, batch_size=args.lstm_batch, time_steps= args.time_steps, LSTM_size= args.LSTM_size, laten_size = args.laten_size)
    def action_decorder(name):
        return  MlpPolicy(name=name, obs_space = obs_space, ac_space = ac_space, embedding_shape = args.laten_size, hid_size = args.pol_hid_size, num_hid_layers = args.pol_layers)
    def state_decorder(name):
        return MlpPolicy_state(name=name, obs_space=obs_space, embedding_shape = args.laten_size, hid_size= args.state_de_hid_size, num_hid_layers = args.state_de_hid_num)
    state_dataset = load_state_dataset(args.state_dir_path, env, args.control_timestep)
    learn(encoder = encoder, action_decorder=action_decorder, state_decorder=state_decorder, embedding_shape= args.laten_size ,dataset=state_dataset, logdir=args.logdir,
          batch_size = args.lstm_batch, time_steps = args.time_steps, epsilon = args.epsilon, lr_rate= args.lr_rate)

def main():
    args = get_arguments()
    train(args)

if __name__ == '__main__':
    main()