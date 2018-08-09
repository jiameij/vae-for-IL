from common import Dataset, explained_variance, fmt_row, zipsame
from model import dataset
from common import logger
import common.tf_util as U
import tensorflow as tf, numpy as np
import time, os, sys
from common.mpi_adam import MpiAdam
from common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import pickle as pkl

# Sample one trajectory (until trajectory end)
def traj_episode_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode

    # Initialize history arrays
    obs = []; rews = []; news = []; acs = []

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)
        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if t > 0 and (new or t % horizon == 0):
            # convert list into numpy array
            obs = np.array(obs)
            rews = np.array(rews)
            news = np.array(news)
            acs = np.array(acs)
            yield {"ob":obs, "rew":rews, "new":news, "ac":acs,
                    "ep_ret":cur_ep_ret, "ep_len":cur_ep_len}
            ob = env.reset()
            cur_ep_ret = 0; cur_ep_len = 0; t = 0

            # Initialize history arrays
            obs = []; rews = []; news = []; acs = []
        t += 1

def traj_segment_generator(pi, env, discriminator, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    cur_ep_true_ret = 0
    ep_true_rets = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_true_rets": ep_true_rets}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = discriminator.get_reward(ob, ac)
        ob, true_rew, new, _ = env.step(ac)
        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, discriminator, expert_dataset,
        timesteps_per_batch,  *,
        g_step, d_step,# timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5, d_stepsize=3e-4,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        save_per_iter=100,
        ckpt_dir=None, task="train",
        sample_stochastic=True,
        load_model_path=None, task_name=None, max_sample_traj=1500
        ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    d_adam = MpiAdam(discriminator.get_trainable_variables())
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    adam.sync()

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, discriminator,timesteps_per_batch, stochastic=True)
    traj_gen = traj_episode_generator(pi, env, timesteps_per_batch, stochastic=sample_stochastic)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=100)

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    if task == 'sample_trajectory':
        # not elegant, i know :(
        sample_trajectory(load_model_path, max_sample_traj, traj_gen, task_name, sample_stochastic)
        sys.exit()

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        # Save model
        if iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            U.save_state(os.path.join(ckpt_dir, task_name), counter=iters_so_far)

        logger.log("********** Iteration %i ************"%iters_so_far)
        for _ in range(g_step):
            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

            assign_old_eq_new() # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
            meanlosses,_,_ = mpi_moments(losses, axis=0)

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, discriminator.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                      include_final_partial_batch=False, batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for discriminator
            if hasattr(discriminator, "obs_rms"): discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = discriminator.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g), d_stepsize)
            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        # ----------------- logger --------------------
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, true_rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        true_rewbuffer.extend(true_rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()


def sample_trajectory(load_model_path, max_sample_traj, traj_gen, task_name, sample_stochastic):

    assert load_model_path is not None
    U.load_state(load_model_path)
    sample_trajs = []
    for iters_so_far in range(max_sample_traj):
        logger.log("********** Iteration %i ************"%iters_so_far)
        traj = traj_gen.__next__()
        ob, new, ep_ret, ac, rew, ep_len = traj['ob'], traj['new'], traj['ep_ret'], traj['ac'], traj['rew'], traj['ep_len']
        logger.record_tabular("ep_ret", ep_ret)
        logger.record_tabular("ep_len", ep_len)
        logger.record_tabular("immediate reward", np.mean(rew))
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
        traj_data = {"ob":ob, "ac":ac, "rew": rew, "ep_ret":ep_ret}
        sample_trajs.append(traj_data)

    sample_ep_rets = [traj["ep_ret"] for traj in sample_trajs]
    logger.log("Average total return: %f"%(sum(sample_ep_rets)/len(sample_ep_rets)))
    if sample_stochastic:
        task_name = 'stochastic.' + task_name
    else:
        task_name = 'deterministic.' + task_name
    pkl.dump(sample_trajs, open(task_name+".pkl", "wb"))


def evaluate(env, policy_func, load_model_path, timesteps_per_batch, number_trajs=10,
             stochastic_policy=False):
    from tqdm import tqdm
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=False)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    ep_gen = traj_episode_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
    U.load_state(load_model_path)

    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = ep_gen.__next__()
        ep_len, ep_ret = traj['ep_len'], traj['ep_ret']
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    print("Average length:", sum(len_list) / len(len_list))
    print("Average return:", sum(ret_list) / len(ret_list))

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
