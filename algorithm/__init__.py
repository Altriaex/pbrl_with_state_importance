import bz2
import os.path as osp
import pickle
import cv2
import numpy as np
import gin
import tensorflow.compat.v1 as tf

import os
from .policy_learning import policy_learning_input_fn
from .preference_learning import train_reward_model, compute_clip_reward, infer_rewards, generate_perturbed_expert_demo_datasets
from .contrastive_learning import train_encoder, compute_pretrain_repre
from ..sampling_utils import enumerate_files
from ..utils import load_logs, atari_observation_masks
from absl import logging
import gc
from PIL import Image
import gym
from .networks import ContinuousObservationEncoder, ObservationEncoder
import sys
try:
    import d4rl
    import glfw
except ImportError as e:
    print('d4rl not found', file=sys.stderr)
    print(e, file=sys.stderr)

def set_preference_path(FLAGS):
    preference_path = FLAGS.exp_path
    for sub_directory in [FLAGS.game, FLAGS.split,
                        "data"]:
        preference_path = osp.join(preference_path, sub_directory)
        if not osp.exists(preference_path):
            os.mkdir(preference_path)
    require_preference_root = ["sample_trajectories", "compute_pretrain_repre", "trajectory_representation_learning_input_fn", "sample_queries", "generate_bt_answers", "reward_learning_input_fn","compute_clip_reward","generate_trex_answers", "sample_ctl_queries", "sample_d4rl_trajectories","sample_ctl_queries2"]
    for param in require_preference_root:
        gin.bind_parameter(param + '.preference_path', preference_path)
    gin.bind_parameter('sample_trajectories.game', FLAGS.game)

def set_action_space(FLAGS):
    if FLAGS.game in atari_observation_masks:
        env = gym.make(FLAGS.game + '-v0')
        n_action = env.action_space.n
        del env
    else:
        n_action = -1
    for param in ['train_policy', 'test_policy', "BT", "LassoBT", "CrowdLassoBT", "CrowdLassoBT2", "CrowdLassoBT3"]:
        gin.bind_parameter(param + '.n_action', n_action)

@gin.configurable
def sample_trajectories(log_path, log_split, game, n_trajectory_per_file, n_clips_per_trajectory, frames_per_clip, preference_path, suffixes=None):
    logging.get_absl_handler().use_absl_log_file('logs', preference_path)
    log_path = osp.join(log_path, game, log_split, "replay_logs")
    if suffixes is None:
        suffix_list, n_file = enumerate_files(log_path)
        logging.info("Found {} log files.".format(n_file))
    else:
        logging.info(f"Sampling from suffixes {suffixes}.")
        suffix_list = suffixes.split(",")
        n_file = len(suffix_list)
    trajectories = []
    tid = 0
    for suffix in suffix_list:
        logging.info(f"Sampling from suffix {suffix}...")
        records = load_logs(log_path, suffix)
        end_of_trajectoreis = np.nonzero(records['terminal'])[0]
        trajectories_in_file = []
        start, end = 0, 0
        for t_count in range(len(end_of_trajectoreis)):
            start, end = end, end_of_trajectoreis[t_count]+1
            trajectories_in_file.append((start, end))
        if len(trajectories_in_file) <= n_trajectory_per_file:
            inds = [i for i in range(len(trajectories_in_file))]
        else:
            inds = np.random.choice(
                len(trajectories_in_file), n_trajectory_per_file, False)
        for ind in inds:
            tstart = trajectories_in_file[ind][0]
            tend = trajectories_in_file[ind][1]
            if tend-tstart-frames_per_clip < 0:
                continue
            elif tend-tstart-frames_per_clip <= n_clips_per_trajectory:
                cstarts = [i for i in range(tend-tstart-frames_per_clip)]
            else:
                cstarts = np.random.choice(
                    tend-tstart-frames_per_clip, n_clips_per_trajectory, False)
            clips = {i: (cstarts[i], cstarts[i]+frames_per_clip)\
                        for i in range(len(cstarts))}
            obs = records['observation'][tstart:tend]*atari_observation_masks[game]
            action = records['action'][tstart:tend]
            reward = records['reward'][tstart:tend]
            terminal = records['terminal'][tstart:tend]
            trajectories.append(dict(tid=tid,
                                     file_suffix=suffix,
                                     start=tstart,
                                     end=tend,
                                     observation=obs,
                                     action=action,
                                     reward=reward,
                                     terminal=terminal,
                                     clips=clips))
            tid += 1
        del records
        gc.collect()
    
    index_base = 0
    obs_array = []
    act_array = []
    starts = []
    tids = []
    trajectories = sorted(trajectories, key=lambda t: int(t["file_suffix"]))
    for trajectory in trajectories:
        for i in range(len(trajectory['observation'])-frames_per_clip):
            starts.append(i+index_base)
            tids.append(trajectory['tid'])
        index_base += len(trajectory['observation'])
        obs_array.append(trajectory['observation'])
        act_array.append(trajectory['action'])
    tids = np.array(tids)
    starts = np.array(starts)
    observations = np.concatenate(obs_array, axis=0)[:,:,:,None]
    actions = np.concatenate(act_array, axis=0)
    np.save(osp.join(preference_path, "tids.npy"), tids)
    np.save(osp.join(preference_path, "starts.npy"), starts)
    np.save(osp.join(preference_path, "observations.npy"), observations)
    np.save(osp.join(preference_path, "actions.npy"), actions)
    del observations, actions
    gc.collect()
    with bz2.BZ2File(osp.join(preference_path, "trajectories.pbz2"), 'w') as f:
        pickle.dump(trajectories, f)

@gin.configurable
def sample_queries(preference_path, n_train_clips, n_query_per_trajectory):
    with bz2.BZ2File(osp.join(preference_path,
                              "trajectories.pbz2"), 'rb') as f:
        trajectories = pickle.load(f)
    queries = []
    for trajectory in trajectories:
        l_tid = trajectory['tid']
        for _ in range(n_query_per_trajectory):
            l_cid = np.random.choice(n_train_clips)
            l_part = (l_tid, l_cid)
            r_part = l_part
            while l_part == r_part\
                or (l_part + r_part) in queries\
                or (r_part + l_part) in queries:
                r_ind = np.random.choice(len(trajectories))
                r_tid = trajectories[r_ind]["tid"]
                r_cid = np.random.choice(n_train_clips)
                r_part = (r_tid, r_cid)
            queries.append(l_part + r_part)
    ret = []
    for l_tid, l_cid, r_tid, r_cid in queries:
        ret.append({"left_tid": l_tid, "left_cid": l_cid,
                    "right_tid": r_tid, "right_cid": r_cid})
    with open(osp.join(preference_path, "queries.pickle"), "wb") as f:
        pickle.dump(ret, f)
    
@gin.configurable
def generate_bt_answers(preference_path, n_task_repeat, n_query_per_task):
    answers = []
    with open(osp.join(preference_path, "queries.pickle"), 'rb') as f:
        queries = pickle.load(f)
    with bz2.BZ2File(osp.join(preference_path,
                              "trajectories.pbz2"), 'rb') as f:
        trajectories = pickle.load(f)
    def find_trajectory(tid):
        for trajectory in trajectories:
            if trajectory['tid'] == tid:
                return trajectory
    n_task = len(queries) // n_query_per_task
    for task_ind in range(n_task):
        for repeat in range(n_task_repeat):
            for query in queries[task_ind * n_query_per_task:\
                                    (task_ind + 1) * n_query_per_task]:
                l_trajectory = find_trajectory(query["left_tid"])
                l_start, l_end = l_trajectory["clips"][query["left_cid"]]
                l_sum_reward = sum(l_trajectory["reward"][l_start: l_end])
                r_trajectory = find_trajectory(query["right_tid"])
                r_start, r_end = r_trajectory["clips"][query["right_cid"]]
                r_sum_reward = sum(r_trajectory["reward"][r_start: r_end])
                max_reward = max([l_sum_reward, r_sum_reward])
                l_score = np.exp(l_sum_reward - max_reward)
                r_score = np.exp(r_sum_reward - max_reward)
                prob = l_score / (l_score + r_score)
                if np.random.random() <= prob:
                    label = 1.0
                else:
                    label = 0.
                query.update({"label": label,
                              "w_id": 0.,
                              "left_start": l_start,
                              "right_start": r_start})
                answers.append(query)
    np.random.shuffle(answers)
    with open(osp.join(preference_path, "bt_answer.pickle"), "wb") as f:
        pickle.dump(answers, f)

@gin.configurable
def generate_trex_answers(preference_path, n_task_repeat):
    answers = []
    with open(osp.join(preference_path, "queries.pickle"), 'rb') as f:
        queries = pickle.load(f)
    with bz2.BZ2File(osp.join(preference_path,
                              "trajectories.pbz2"), 'rb') as f:
        trajectories = pickle.load(f)
    def find_trajectory(tid):
        for trajectory in trajectories:
            if trajectory['tid'] == tid:
                return trajectory
    for query in queries:
        for repeat in range(n_task_repeat):
            l_trajectory = find_trajectory(query["left_tid"])
            r_trajectory = find_trajectory(query["right_tid"])
            l_start = l_trajectory["clips"][query["left_cid"]][0]
            r_start = r_trajectory["clips"][query["right_cid"]][0]
            l_sum_reward = sum(l_trajectory["reward"])
            r_sum_reward = sum(r_trajectory["reward"])
            max_reward = max([l_sum_reward, r_sum_reward])
            l_score = np.exp(l_sum_reward - max_reward)
            r_score = np.exp(r_sum_reward - max_reward)
            prob = l_score / (l_score + r_score)
            if np.random.random() <= prob:
                label = 1.0
            else:
                label = 0.
            query.update({"label": label,
                          "w_id": 0.,
                          "left_start": l_start,
                          "right_start": r_start})
            answers.append(query)
    np.random.shuffle(answers)
    with open(osp.join(preference_path, "trex_answer.pickle"), "wb") as f:
        pickle.dump(answers, f)

@gin.configurable
def sample_ctl_queries(FLAGS, preference_path, n_queries, moment_len, frames_per_clip):
    with bz2.BZ2File(osp.join(preference_path,
                              "trajectories.pbz2"), 'rb') as f:
        trajectories = pickle.load(f)
    model_path = osp.join(
        FLAGS.exp_path, FLAGS.game, FLAGS.split, "lasso_bt3")
    lbt3_weights = np.squeeze(
        np.load(osp.join(model_path, "frame_weights.npy")))
    left_len = moment_len // 2
    right_len = moment_len - left_len
    mid_index = np.argmax(lbt3_weights,axis=1)
    starts = np.maximum(mid_index -left_len, 0)
    ends = np.minimum(mid_index + right_len, frames_per_clip)
    ind = 0
    moments = []
    for trajectory in trajectories:
        for cid, (clip_start, clip_end) in trajectory["clips"].items():
            obs = trajectory["observation"][clip_start:clip_end]
            imp_obs = obs[starts[ind]:ends[ind]]
            moments.append(imp_obs)
            ind += 1
    sampled = set()
    queries = []
    while len(queries) < n_queries:
        pair = np.random.choice(len(moments), 2)
        while (pair[0], pair[1]) in sampled or (pair[1], pair[0]) in sampled:
            pair = np.random.choice(len(moments), 2)
        sampled.add((pair[0], pair[1]))
        queries.append(
            dict(left_moment=moments[pair[0]], right_moment=moments[pair[1]]))
    with open(osp.join(model_path, "ctl_queries.pickle"), "wb") as f:
        pickle.dump(queries, f)
    gif_dir = osp.join(model_path, "ctl_query_gifs")
    os.makedirs(gif_dir, exist_ok=True)
    for pid, pair in enumerate(queries):
        arr = pair['left_moment']
        imgs = [Image.fromarray(img) for img in arr]
        imgs[0].save(osp.join(gif_dir, f"{pid:04d}_0.gif"), save_all=True,
                     append_images=imgs[1:], duration=200, loop=0)
        arr = pair['right_moment']
        imgs = [Image.fromarray(img) for img in arr]
        imgs[0].save(osp.join(gif_dir, f"{pid:04d}_1.gif"), save_all=True,
                     append_images=imgs[1:], duration=200, loop=0)

@gin.configurable
def sample_ctl_queries2(FLAGS, preference_path, moment_len, frames_per_clip,d4rl_render_shape=(256,256)):
    with bz2.BZ2File(osp.join(preference_path,
                              "trajectories.pbz2"), 'rb') as f:
        trajectories = pickle.load(f)
    model_path = osp.join(
        FLAGS.exp_path, FLAGS.game, FLAGS.split, "lasso_bt3")
    lbt3_weights = np.squeeze(
        np.load(osp.join(model_path, "frame_weights.npy")))
    left_len = moment_len // 2
    right_len = moment_len - left_len
    mid_index = np.argmax(lbt3_weights,axis=1)
    starts = np.maximum(mid_index -left_len, 0)
    ends = np.minimum(mid_index + right_len, frames_per_clip)
    ind = 0
    moments = []
    if not FLAGS.game in atari_observation_masks:
        qposes, qvals = [], []
        env = gym.make(FLAGS.game)
        env.reset()
    for trajectory in trajectories:
        for _, (clip_start, clip_end) in trajectory["clips"].items():
            obs = trajectory["observation"][clip_start:clip_end]
            imp_obs = obs[starts[ind]:ends[ind]]
            moments.append(imp_obs)
            if not FLAGS.game in atari_observation_masks:
                qpos = trajectory['infos/qpos']
                qval = trajectory['infos/qvel']
                qposes.append(qpos[starts[ind]:ends[ind]])
                qvals.append(qval[starts[ind]:ends[ind]])
            ind += 1
    with open(osp.join(model_path, "ctl_queries2.pickle"), "wb") as f:
        pickle.dump(moments, f)
    gif_dir = osp.join(model_path, "ctl_query_gifs2")
    os.makedirs(gif_dir, exist_ok=True)
    
    for mid, moment in enumerate(moments):
        if FLAGS.game in atari_observation_masks:
            imgs = [Image.fromarray(img) for img in moment]
        else:
            qpos, qval = qposes[mid], qvals[mid]
            imgs = []
            for i in range(len(moment)):
                env.set_state(qpos[i], qval[i])
                img = env.render(mode="rgb_array")
                img = cv2.resize(img, d4rl_render_shape)
                imgs.append(Image.fromarray(img))
                #glfw.terminate()  
        imgs[0].save(osp.join(gif_dir, f"{mid:04d}.gif"), save_all=True,
                     append_images=imgs[1:], duration=50, loop=0)

@gin.configurable
def sample_d4rl_trajectories(game, n_d4rl_trajecotry, n_clips_per_trajectory, frames_per_clip, preference_path):
    env = gym.make(game)
    dataset = env.get_dataset()
    by_terminals = np.nonzero(dataset["terminals"].astype(np.int32))[0]
    by_timeouts = np.nonzero(dataset["timeouts"].astype(np.int32))[0]
    # separate trajectories by terminals or timeouts
    if len(by_terminals) > 100:
        end_of_trajectoreis = by_terminals
    elif len(by_timeouts) > 100:
        end_of_trajectoreis = by_timeouts
    else:
        raise NotImplementedError
    trajectory_lens = end_of_trajectoreis[1:] - end_of_trajectoreis[:-1]
    mean_lens = np.mean(trajectory_lens)
    if mean_lens < frames_per_clip:
        ratio = int(frames_per_clip/mean_lens) + 1
        new_end_of_trajectories = end_of_trajectoreis[::ratio]
        end_of_trajectoreis = new_end_of_trajectories
    trajectories_in_file = []
    start, end = 0, 0
    trajectories = []
    tid = 0
    for t_count in range(len(end_of_trajectoreis)):
        start, end = end, end_of_trajectoreis[t_count]+1
        if end > start + 2*frames_per_clip:
            trajectories_in_file.append((start, end))
    if len(trajectories_in_file) > n_d4rl_trajecotry:
        inds = np.random.choice(
                len(trajectories_in_file), n_d4rl_trajecotry, False)
    else:
        inds = [i for i in range(len(trajectories_in_file))]
    for ind in inds:
        tstart = trajectories_in_file[ind][0]
        tend = trajectories_in_file[ind][1]
        cstarts = np.random.choice(
            tend-tstart-frames_per_clip, n_clips_per_trajectory, False)
        clips = {i: (cstarts[i], cstarts[i]+frames_per_clip)\
                    for i in range(len(cstarts))}
        trajectory = {}
        for key in dataset.keys():
            if key.startswith("metadata"):
                continue
            if key in ["observations", "actions", "rewards"]:
                left_key = key[:-1]
            else:
                left_key = key
            trajectory[left_key] = dataset[key][tstart:tend]
        trajectory["tid"] = tid
        trajectory["start"] = tstart
        trajectory["end"] = tend
        trajectory["clips"] = clips
        trajectories.append(trajectory)
        tid += 1
    trajectories = sorted(trajectories, key=lambda t: int(t["start"]))
    index_base = 0
    obs_array = []
    act_array = []
    starts = []
    tids = []
    for trajectory in trajectories:
        for i in range(len(trajectory['observation'])-frames_per_clip):
            starts.append(i+index_base)
            tids.append(trajectory['tid'])
        index_base += len(trajectory['observation'])
        obs_array.append(trajectory['observation'])
        act_array.append(trajectory['action'])
    tids = np.array(tids)
    starts = np.array(starts)
    observations = np.concatenate(obs_array, axis=0)
    actions = np.concatenate(act_array, axis=0)
    np.save(osp.join(preference_path, "tids.npy"), tids)
    np.save(osp.join(preference_path, "starts.npy"), starts)
    np.save(osp.join(preference_path, "observations.npy"), observations)
    np.save(osp.join(preference_path, "actions.npy"), actions)
    with bz2.BZ2File(osp.join(preference_path, "trajectories.pbz2"), 'w') as f:
        pickle.dump(trajectories, f)