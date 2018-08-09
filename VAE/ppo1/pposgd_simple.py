from common import Dataset, explained_variance, fmt_row, zipsame
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

def traj_segment_generator(pi, timesteps ,env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    ob_reset = ob.copy()
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon+1)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ob_s = np.array([ob_reset for _ in range(timesteps)])
        if(t < timesteps):
            # ob_s = []
            # for i in range(timesteps-t -1):
            #     ob_s.append(ob_reset)
            if(t > 0):
                for _ in range(t):
                    ob_s[timesteps+_-1-t] = obs[_]

         #   obs = np.array(list(ob_s).extend(list(obs[:t+1])))
         #   ob_s.extend(obs[:t+1])
          #  ac, vperd = pi.act(stochastic, ob_s)
        else:
            ob_s = obs[t+1-timesteps:t+1]
        ob_s[timesteps - 1] = ob
        ac, vpred = pi.act(stochastic, ob_s)
        # ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs[:horizon], "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ob_reset":ob_reset}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            t = 0
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
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

def learn(env, policy_func, *,
        timesteps = 4,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        save_per_iter=100,
        ckpt_dir=None, task="train",
        sample_stochastic=True,
        load_model_path=None, task_name=None, max_sample_traj=1500
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", timesteps, ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", timesteps, ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    pi_vpred = tf.placeholder(dtype=tf.float32, shape=[None])
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
#    ob_now = tf.placeholder(dtype=tf.float32, shape=[optim_batchsize, list(ob_space.shape)[0]])
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
   # total_loss = pol_surr + pol_entpen + vf_loss
    total_loss = pol_surr + pol_entpen
    losses = [pol_surr, pol_entpen, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith("vf")]
    pol_var_list = [v for v in var_list if not v.name.split("/")[1].startswith("vf")]
  #  lossandgrad = U.function([ob, ac, atarg ,ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, pol_var_list)])
    vf_grad = U.function([ob, ac, atarg, ret, lrmult], U.flatgrad(vf_loss, vf_var_list))

    # adam = MpiAdam(var_list, epsilon=adam_epsilon)
    pol_adam = MpiAdam(pol_var_list, epsilon=adam_epsilon)
    vf_adam = MpiAdam(vf_var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    #adam.sync()
    pol_adam.sync()
    vf_adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, timesteps, env, timesteps_per_batch, stochastic=True)
    traj_gen = traj_episode_generator(pi, env, timesteps_per_batch, stochastic=sample_stochastic)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    EpRewMean_MAX = 2.5e3
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
        # if(iters_so_far == 1):
        #     a = 1
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg,vpred, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["vpred"],seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vpred = vpred, vtarg=tdlamret), shuffle=False) #d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vpred = vpred, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            pre_obs = [seg["ob_reset"] for jmj in range(timesteps - 1)]
            for batch in d.iterate_once(optim_batchsize):
                ##feed ob, 重新处理一下ob,在batch["ob"]的最前面插入timesteps-1个env.reset的ob,然后滑动串口划分一下batch['ob]
                ob_now = np.append(pre_obs , batch['ob']).reshape(optim_batchsize+timesteps-1, list(ob_space.shape)[0])
                pre_obs = ob_now[-(timesteps-1):]
                ob_fin = []
                for jmj in range(optim_batchsize):
                    ob_fin.append(ob_now[jmj:jmj+timesteps])
                *newlosses, g = lossandgrad(ob_fin, batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult) ###这里的g好像都是0
                #adam.update(g, optim_stepsize * cur_lrmult)
                pol_adam.update(g, optim_stepsize * cur_lrmult)
                vf_g = vf_grad(ob_fin, batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                vf_adam.update(vf_g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

            pre_obs = [seg["ob_reset"] for jmj in range(timesteps - 1)]
            for batch in d.iterate_once(optim_batchsize):
                ##feed ob, 重新处理一下ob,在batch["ob"]的最前面插入timesteps-1个env.reset的ob,然后滑动串口划分一下batch['ob]
                ob_now = np.append(pre_obs, batch['ob']).reshape(optim_batchsize+timesteps-1, list(ob_space.shape)[0])
                pre_obs = ob_now[-(timesteps - 1):]
                ob_fin = []
                for jmj in range(optim_batchsize):
                    ob_fin.append(ob_now[jmj:jmj+timesteps])
                *newlosses, g = lossandgrad(ob_fin, batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult) ###这里的g好像都是0
                #adam.update(g, optim_stepsize * cur_lrmult)
                pol_adam.update(g, optim_stepsize * cur_lrmult)
                vf_g = vf_grad(ob_fin, batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                vf_adam.update(vf_g, optim_stepsize * cur_lrmult)

        logger.log("Evaluating losses...")
        losses = []
        loss_pre_obs = [seg["ob_reset"] for jmj in range(timesteps-1)]
        for batch in d.iterate_once(optim_batchsize):
            ### feed ob
            ob_now = np.append(loss_pre_obs , batch['ob']).reshape(optim_batchsize+timesteps-1, list(ob_space.shape)[0])
            loss_pre_obs = ob_now[-(timesteps-1):]
            ob_fin = []
            for jmj in range(optim_batchsize):
                ob_fin.append(ob_now[jmj:jmj+timesteps])
            newlosses = compute_losses(ob_fin, batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        if(np.mean(rewbuffer) > EpRewMean_MAX):
            EpRewMean_MAX = np.mean(rewbuffer)
            print(iters_so_far)
            print(np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
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

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
