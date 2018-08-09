import argparse
from common import set_global_seeds, tf_util as U
import gym, logging, sys
import bench
import os.path as osp
from common import logger
import tensorflow as tf
from dataset.mujoco import Mujoco_Dset
import numpy as np
import ipdb

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Humanoid-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=3)
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=1)
    parser.add_argument('--expert_path', type=str, default='stochastic.Humanoid-v2.pkl')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], default='train')

    # for encoder
    parser.add_argument('--LSTM_size', type=int, default=300, help="bi_directional LSTM's hidden size")
    parser.add_argument('--laten_size', type=int, default=30, help="the size of laten vector Z")
    parser.add_argument('--time_steps', type=int, default=4, help="how many obs to lstm")
    parser.add_argument('--lstm_batch', type=int, default=1, help="lstm batch size")
    parser.add_argument('--encoder_load_path', type = str, default=None)

    # for demonstrations
    parser.add_argument('--expert_data_dir', type=str, default="/home/jmj/08/")
    parser.add_argument('--control_timestep', type=float, default=0.01)


    # for evaluatation
    parser.add_argument('--stochastic_policy', type=bool, default=True)
    #  Mujoco Dataset Configuration
    parser.add_argument('--ret_threshold', help='the return threshold for the expert trajectories', type=int, default=0)
    parser.add_argument('--traj_limitation', type=int, default=np.inf)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=list, default=[300, 200])
    parser.add_argument('--adversary_hidden_layers', type=int, default=2)
    parser.add_argument("--adversary_learning_rate", type=float, default=1e-4)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['bc', 'trpo', 'ppo'], default='ppo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    parser.add_argument('--pretrained', help='Use BC to pretrain', type=bool, default=False)
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()

def get_task_name(args):
    if args.algo == 'bc':
        task_name = 'behavior_cloning.'
        if args.traj_limitation != np.inf: task_name += "traj_limitation_%d."%args.traj_limitation
        task_name += args.env_id.split("-")[0]
    else:
        task_name = args.algo + "_gail."
        if args.pretrained: task_name += "with_pretrained."
        if args.traj_limitation != np.inf: task_name += "traj_limitation_%d."%args.traj_limitation
        task_name += args.env_id.split("-")[0]
        if args.ret_threshold > 0: task_name += ".return_threshold_%d" % args.ret_threshold
        task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
                ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    return task_name

def load(saver, sess, logdir): ## for load encoder
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


def main(args):
    from ppo1 import mlp_policy ##for policy
    from model.encoder import bi_direction_lstm
    from dm_control.suite import humanoid_CMU

    U.make_session(num_cpu=args.num_cpu).__enter__()
    set_global_seeds(args.seed)
    env = humanoid_CMU.stand()
    obs_space = env.physics.data.qpos
    ac_space = env.action_spec()
    def policy_fn(name, ob_space, ac_space, reuse=False): ###mlp policy 要不要用用之前训好的policy,不是的
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            reuse=reuse, hid_size= [300, 200, 100], num_hid_layers=3)
    def encoder(name):
        return bi_direction_lstm(name=name, obs_space=obs_space, batch_size=args.lstm_batch, time_steps= args.time_steps, LSTM_size= args.LSTM_size, laten_size = args.laten_size)

    lstm_encoder = encoder("lstm_encoder")
    saver = lstm_encoder.get_trainable_variables()
    load(saver=saver, sess=tf.get_default_session(), logdir = args.encoder_load_path) ###将encoder的参数load进去
    # env = bench.Monitor(env, logger.get_dir() and
    #     osp.join(logger.get_dir(), "monitor.json"))
    # env.seed(args.seed)
    # gym.logger.setLevel(logging.WARN)
    # task_name = get_task_name(args)
    task_name = "Humanoid-CMU"
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    # dataset = Mujoco_Dset(expert_path=args.expert_path, ret_threshold=args.ret_thres hold, traj_limitation=args.traj_limitation)
    # ================ Sample trajectory τj from the demonstration ============================= # 相当于expert dataset,仅需要obs即可
    from model.VAE import load_state_dataset
    dataset = load_state_dataset(data_dir_path=args.expert_data_dir, env = env, control_timestep=args.control_timestep)

    pretrained_weight = None
    if (args.pretrained and args.task == 'train') or args.algo == 'bc':
        # Pretrain with behavior cloning
        from gail import behavior_clone
        if args.algo == 'bc' and args.task == 'evaluate':
            behavior_clone.evaluate(env, policy_fn, args.load_model_path, stochastic_policy=args.stochastic_policy)
            sys.exit()
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
            max_iters=args.BC_max_iter, pretrained=args.pretrained,
            ckpt_dir=args.checkpoint_dir, log_dir=args.log_dir, task_name=task_name)
        if args.algo == 'bc':
            sys.exit()

    from network.adversary import TransitionClassifier
    # discriminator
    discriminator = TransitionClassifier(env, args.adversary_hidden_size, hidden_layers = args.adversary_hidden_layers, lr_rate = args.adversary_learning_rate, entcoeff=args.adversary_entcoeff, embedding_shape=args.laten_size) ###embedding_z，现在还没有处理
    observations = dataset.get_next_batch(batch_size=128)[0].transpose((1, 0))   ### !!!!这个地方还是稍微有点儿乱啊
    embedding_z = lstm_encoder.get_laten_vector(observations)
    if args.algo == 'trpo':
        # Set up for MPI seed
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        from gail import trpo_mpi
        if args.task == 'train':
            trpo_mpi.learn(env, policy_fn, discriminator, dataset, embedding_z=None, ##embedding_z这里现在我还没有想好
                pretrained=args.pretrained, pretrained_weight=pretrained_weight,
                g_step=args.g_step, d_step=args.d_step,
                timesteps_per_batch=1024,
                max_kl=args.max_kl, cg_iters=10, cg_damping=0.1,
                max_timesteps=args.num_timesteps,
                entcoeff=args.policy_entcoeff, gamma=0.995, lam=0.97,
                vf_iters=5, vf_stepsize=1e-3,
                ckpt_dir=args.checkpoint_dir, log_dir=args.log_dir,
                save_per_iter=args.save_per_iter, load_model_path=args.load_model_path,
                task_name=task_name)
        elif args.task == 'evaluate':
            trpo_mpi.evaluate(env, policy_fn, args.load_model_path, timesteps_per_batch=1024,
                number_trajs=10, stochastic_policy=args.stochastic_policy)
        else: raise NotImplementedError
    elif args.algo == 'ppo':
        # Set up for MPI seed
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        from gail import ppo_mpi
        if args.task == 'train':
            ppo_mpi.learn(env, policy_fn, discriminator, dataset,
                           # pretrained=args.pretrained, pretrained_weight=pretrained_weight,
                           timesteps_per_batch=1024,
                           g_step=args.g_step, d_step=args.d_step,
                           # max_kl=args.max_kl, cg_iters=10, cg_damping=0.1,
                           clip_param= 0.2,entcoeff=args.policy_entcoeff,
                           max_timesteps=args.num_timesteps,
                            gamma=0.99, lam=0.95,
                           # vf_iters=5, vf_stepsize=1e-3,
                            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                          d_stepsize=3e-4,
                          schedule='linear', ckpt_dir=args.checkpoint_dir,
                          save_per_iter=100, task=args.task,
                          sample_stochastic=args.stochastic_policy,
                          load_model_path=args.load_model_path,
                          task_name=task_name)
        elif args.task == 'evaluate':
            ppo_mpi.evaluate(env, policy_fn, args.load_model_path, timesteps_per_batch=1024,
                              number_trajs=10, stochastic_policy=args.stochastic_policy)
        else:
            raise NotImplementedError
    else: raise NotImplementedError

    env.close()

if __name__ == '__main__':
    args = argsparser()
    main(args)
