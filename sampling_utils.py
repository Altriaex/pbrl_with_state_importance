'''
Description: 
Author: CoeusZhang
Date: 2021-05-03 18:08:55
LastEditTime: 2022-01-21 17:42:56
'''
import bz2
import os
import os.path as osp
import pickle
from itertools import chain
import numpy as np
import gc

from absl import logging
from .utils import load_logs
import gin
from collections import defaultdict

# Set random seed

def enumerate_files(log_path):
    '''
    @Description: Enumerate logs of a game. The enire dataset is very large, so
        we use a few log files in development. This function is for enumerating
        available logs.
    @Param: 
        log_path: A str for path to logs of some game.
            e.g. <dataset_path>/<game>/<split>
    @Return: 
        parts_suffix_map: A dict for available suffix for the game. 
        n_files: A int for number of files available.
    '''
    possible_suffix = set([str(s) for s in range(0, 51)])
    available_suffix = set([s.split(".")[-2] for s in os.listdir(log_path)])
    available_suffix = available_suffix & possible_suffix
    # path to logs: <dataset_path>/<game>/<part>/replay_logs/
    # For each game, there are 51 parts in the original dataset. Each part
    # consists of several sub-parts, which can be identified with suffixes.
    available_suffix = sorted(list(available_suffix))
    return available_suffix, len(available_suffix)

def draw_clips_from_records(records, n_clip, frames_per_clip):
    '''
    @Description: Sample clips from records loaded from game logs.
    @Param: 
        records: A dict with keys: observation, action, reward, terminal
        n_clip: An int for the number of clips to be sampled.
        frames_per_clip: An int for the number of frames in a clip.
    @Return: 
        clips: A list containing dicts for clips. Each dict has four keys: 
            start_ind, end_ind, reward, observation, action. The first two
            fields are for indices for the clip in this record. The reward field
            is an int for the sum of rewards of the clip. The last two fields
            are numpy arrays for observations and actions of this clip.
    '''
    # Extract indices for starting frame and ending frame of
            # trajectories.
    end_of_trajectoreis = np.nonzero(records['terminal'])[0]
    trajectories = []
    start, end = 0, 0
    for t_count in range(len(end_of_trajectoreis)):
        start, end = end, end_of_trajectoreis[t_count]+1
        trajectories.append((start, end))
    clips = []
    # Draw a trajectory, then randomly clip it
    for _ in range(n_clip):
        start, end = 0, 0
        while end - start < frames_per_clip:
            t_id = np.random.randint(0, len(trajectories))
            start, end = trajectories[t_id]
        end -= frames_per_clip
        start = np.random.randint(start, end)
        end = start + frames_per_clip
        reward = records['reward'][start: end].copy()
        obs = records['observation'][start: end].copy()
        action = records['action'][start: end].copy()
        clips.append(dict(start_ind=start, end_ind=end, reward=reward,
                            observation=obs, action=action))
    return clips

@gin.configurable
def sample_clips(log_path, log_split, game, n_clips, frames_per_clip, preference_path, suffixes=None):
    log_path = osp.join(log_path, game, log_split, "replay_logs")
    if suffixes is None:
        suffix_list, n_file = enumerate_files(log_path)
        logging.info("Found {} log files.".format(n_file))
    else:
        logging.info(f"Sampling from suffixes {suffixes}.")
        suffix_list = suffixes.split(",")
        n_file = len(suffix_list)
    clips_per_file = n_clips // n_file
    n_remain = n_clips - clips_per_file * n_file
    n_sampled = 0
    sample_queries = {}
    for suffix in suffix_list:
        records = load_logs(log_path, suffix)
        clips = draw_clips_from_records(records, clips_per_file,
                                            frames_per_clip)
        logging.info(f"Sampled {len(clips)} clips from suffix {suffix}...")
        n_sampled += len(clips)
        sample_queries[suffix] = clips
        del records
        gc.collect()
    n_remain = n_clips - n_sampled
    if n_remain > 0:
        records = load_logs(log_path, suffix)
        remain_clip = draw_clips_from_records(records, n_remain, frames_per_clip)
        logging.info(f"Sampled {len(remain_clip)} clips from suffix {suffix}...")
        sample_queries[suffix].extend(remain_clip)
        del records
        gc.collect()
    logging.info("Saving clips to disk...") 
    with bz2.BZ2File(osp.join(preference_path, "clips.pbz2"), 'w') as f:
        pickle.dump(sample_queries, f)  

@gin.configurable
def generate_synthetic_workers(reliability_alpha, reliability_beta, n_workers, preference_path):
    if reliability_alpha == 0 and reliability_beta == 0:
        logging.info("Generating perfect workers...")
        reliability = np.ones(n_workers)
    else:
        logging.info("Generating imperfect workers using Beta distribution...") 
        reliability = np.random.beta(reliability_alpha, reliability_beta, size=n_workers)
    np.save(osp.join(preference_path, "synthetic_workers.npy"), reliability)

def get_reliability_generator(reliability, preference_per_worker):
        worker_id = 0
        preference_count = 0
        while True:
            if worker_id == len(reliability) - 1 \
                and preference_count==preference_per_worker:
                yield reliability[-1]
            else:
                if preference_count < preference_per_worker:
                    preference_count += 1
                else:
                    preference_count = 1
                    worker_id += 1
                yield reliability[worker_id], worker_id

def get_reliability_generator2(reliability, label_per_worker):
    worker_jobs = {i:0 for i in range(len(reliability))}
    while True:
        remain_workers = [k for k,v in worker_jobs.items() if v < label_per_worker]
        if len(remain_workers) == 1:
            worker_id = remain_workers[0]
        elif len(remain_workers) > 1:
            worker_id = remain_workers[np.random.randint(low=0, high=len(remain_workers))]
        else:
            raise NotImplementedError
        worker_jobs[worker_id] += 1
        yield reliability[worker_id], worker_id
                
@gin.configurable
def generate_synthetic_preferences(n_preferences, preference_path, label_per_query, label_per_worker):
    logging.info("Loading workers...") 
    reliability = np.load(osp.join(preference_path, "synthetic_workers.npy"))
    n_query = n_preferences // label_per_query
    assert label_per_worker * len(reliability) >= n_preferences
    logging.info("Loading clips...") 
    with bz2.BZ2File(osp.join(preference_path, "clips.pbz2"), 'rb') as f:
        clips = pickle.load(f)
    gen = get_reliability_generator2(reliability, label_per_worker)
    clips = list(chain(*[val for _,val in clips.items()]))
    for part in range(1, 11):
        logging.info("Sampling preference part {}...".format(part))
        preferences = []
        for _ in range(n_query//10):
            c_id1 = np.random.randint(0, len(clips))
            c_id2 = np.random.randint(0, len(clips))
            while c_id2 == c_id1:
                c_id2 = np.random.randint(0, len(clips))
            c1, c2 = clips[c_id1], clips[c_id2]
            reward_sum1, reward_sum2 = sum(c1['reward']), sum(c2['reward'])
            if reward_sum1 == reward_sum2:
                true_label = 0.5
            else:
                true_label = 1 if reward_sum1 > reward_sum2 else 0
            for _ in range(label_per_query):
                label = true_label
                reliability, worker_id = next(gen)
                if np.random.rand() > reliability:
                    # The worker gives incorrect label
                    candidate = [l for l in [1, 0.5, 0] if not l == label]
                    if np.random.rand() < 0.5:
                        label = candidate[0]
                    else:
                        label = candidate[1]
                preferences.append({"observation1": c1['observation'],
                                    "observation2": c2['observation'],
                                    "action1": c1["action"],
                                    "action2": c2["action"],
                                    "reward1": c1["reward"],
                                    "reward2": c2["reward"],
                                    "label": label,
                                    "worker_id": worker_id,
                                    "clip_id1": c_id1,
                                    "clip_id2": c_id2})
        with bz2.BZ2File(osp.join(preference_path,
            "synthetic_preference_{}.pbz2".format(part)), 'w') as f:
            pickle.dump(np.array(preferences), f)

@gin.configurable
def generate_features(preference_path, label_per_query):
    better, equal = defaultdict(list), defaultdict(list)
    worse = defaultdict(list)
    for p in range(1, 11):
        logging.info("Processing preference part {}...".format(p))
        f = osp.join(preference_path, "synthetic_preference_{}.pbz2".format(p))
        with bz2.BZ2File(f, "rb") as f:
            pairs = pickle.load(f)
        for p in pairs:
            key = p["clip_id1"], p["clip_id2"]
            if p["label"] == 1:
                better[key].append(p["worker_id"])
            elif p["label"] == 0.5:
                equal[key].append(p["worker_id"])
            else:
                worse[key].append(p["worker_id"])
        del pairs
        gc.collect()
    for p in range(1, 11):
        logging.info("Writing preference part {}...".format(p))
        f = osp.join(preference_path, "synthetic_preference_{}.pbz2".format(p))
        with bz2.BZ2File(f, "rb") as f:
            pairs = pickle.load(f)
        for pair in pairs:
            key = pair["clip_id1"], pair["clip_id2"]
            w = [-1] * label_per_query
            for ind, worker_id in enumerate(worse[key]):
                if ind < len(w):
                    w[ind] = worker_id
            pair["label_worse"] = w
            e = [-1] * label_per_query
            for ind, worker_id in enumerate(equal[key]):
                if ind < len(e):
                    e[ind] = worker_id
            pair["label_equal"] = e
            b = [-1] * label_per_query
            for ind, worker_id in enumerate(better[key]):
                if ind < len(b):
                    b[ind] = worker_id
            pair["label_better"] = b
        f = osp.join(preference_path, "synthetic_preference_{}.pbz2".format(p))
        with bz2.BZ2File(f, "w") as f:
            pickle.dump(pairs, f)
        del pairs
        gc.collect()
