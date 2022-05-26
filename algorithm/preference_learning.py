import bz2
import pickle
import numpy as np
import os
import os.path as osp
import gin
from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_UNSPECIFIED
import tensorflow.compat.v1 as tf
from ..dummy_wrapped_buffer import DummyWrappedBuffer
from ..sampling_utils import enumerate_files
from absl import logging
import  gc
from ..utils import atari_observation_masks
from . preference_models import BT, CrowdLassoBT2, LassoBT, CrowdLassoBT, CrowdLassoBT3
import gym
import sys
import h5py
try:
    import d4rl
except ImportError as e:
    print('d4rl not found', file=sys.stderr)
    print(e, file=sys.stderr)

def load_ctl_data(preference_path):
    game_spit, _ = osp.split(preference_path)
    game, _ = osp.split(game_spit)
    ctl_path = osp.join(game, "1", "lasso_bt3")
    with open(osp.join(ctl_path, "ctl_queries.pickle"), "rb") as f:
        ctl_queries = pickle.load(f)
    with open(osp.join(ctl_path, "ctl_labels.pickle"), "rb") as f:
        ctl_labels = pickle.load(f)
    while True:
        np.random.shuffle(ctl_labels)
        for annotation in ctl_labels:
            qid = annotation[0]
            left_moment = ctl_queries[qid]["left_moment"]
            right_moment = ctl_queries[qid]["right_moment"]
            label = annotation[1]
            yield {"left_moment": left_moment[:,:,:,None],
                    "right_moment": right_moment[:,:,:,None],
                    "ctl_label": label}

def load_ctl2_data(preference_path, moment_len):
    game_spit, _ = osp.split(preference_path)
    game, _ = osp.split(game_spit)
    ctl_path = osp.join(game, "1", "lasso_bt3")
    with open(osp.join(ctl_path, "ctl_queries2.pickle"), "rb") as f:
        ctl_queries = pickle.load(f)
    with open(osp.join(ctl_path, "ctl_labels2.pickle"), "rb") as f:
        ctl_labels = pickle.load(f)
    
    while True:
        np.random.shuffle(ctl_labels)
        for annotation in ctl_labels:
            qid = annotation[0]
            moment = ctl_queries[qid]
            padding = np.zeros_like(moment[0])
            is_padding = np.ones(moment_len, dtype=np.float32)
            if len(moment) < moment_len:
                padding = np.array([padding] * (moment_len - len(moment)))
                is_padding[len(moment):] = 1.
                moment = np.concatenate([moment, padding], axis=0)
            label = annotation[1]
            yield {"moment": moment[:,:,:,None],
                    "ctl_label": label,
                    "is_ctl_padding": is_padding}

@gin.configurable
def reward_learning_input_fn(preference_path, frames_per_clip, frame_shape, batchsize, split, answer_type, moment_len):
    with bz2.BZ2File(osp.join(preference_path,
                              "trajectories.pbz2"), 'rb') as f:
        trajectories = pickle.load(f)
    tid_base_ind = {}
    base_ind = 0
    for trajectory in trajectories:
        if not trajectory["tid"] in tid_base_ind:
            tid_base_ind[trajectory["tid"]] = base_ind
            base_ind += len(trajectory["observation"])
    del trajectories
    gc.collect()
    trajectory_obs = np.load(osp.join(preference_path, "observations.npy"))
    trajectory_act = np.load(osp.join(preference_path, "actions.npy"))

    def load_answer():
        if answer_type in ["bt", "ctl", "ctl2"]:
            with open(osp.join(preference_path, "bt_answer.pickle"), 'rb') as f:
                answers = pickle.load(f)
            if split == "train":
                answers = answers[int(len(answers)*0.1):]
            if answer_type == "ctl":
                ctl_loader = load_ctl_data(preference_path)
            if answer_type == "ctl2":
                ctl_loader = load_ctl2_data(preference_path, moment_len)
        elif answer_type == "trex":
            with open(osp.join(preference_path, "trex_answer.pickle"), 'rb') as f:
                answers = pickle.load(f)
        else:
            raise NotImplementedError
        while True:
            np.random.shuffle(answers)
            for answer in answers:
                l_start = answer["left_start"] + tid_base_ind[answer["left_tid"]]
                l_end = l_start + frames_per_clip
                l_obs = trajectory_obs[l_start: l_end]
                l_act = trajectory_act[l_start: l_end]
                r_start = answer["right_start"] + tid_base_ind[answer["right_tid"]]
                r_end = r_start + frames_per_clip
                r_obs = trajectory_obs[r_start: r_end]
                r_act = trajectory_act[r_start: r_end]
                ans = {"left_observation": l_obs,
                    "left_action": l_act,
                    "right_observation": r_obs,
                    "right_action": r_act,
                    "label": answer["label"],
                    "left_tid": answer["left_tid"],
                    "right_tid": answer["right_tid"]}
                if answer_type in ["ctl", "ctl2"]:
                    ctl_sample = next(ctl_loader)
                    ans.update(ctl_sample)
                yield ans
    obs_dtype = trajectory_obs.dtype
    obs_shape = trajectory_obs.shape[1:]
    action_dtype = trajectory_act.dtype
    if obs_dtype == np.uint8:
        obs_dtype = tf.uint8
    elif obs_dtype == np.float32:
        obs_dtype = tf.float32
    else:
        raise NotImplementedError
    if action_dtype == np.int32:
        action_dtype = tf.int32
    elif action_dtype == np.float32:
        action_dtype = tf.float32
    else:
        raise NotImplementedError
    if len(trajectory_act.shape) > 1:
        action_shape = trajectory_act.shape[1:]
    else:
        action_shape = ()
    

    output_types = {
            "left_observation": obs_dtype,
            "right_observation": obs_dtype,
            "label": tf.float32,
            "left_tid": tf.int32,
            "right_tid": tf.int32,
            "left_action": action_dtype,
            "right_action": action_dtype}
    output_shapes = {
            "left_observation": tf.TensorShape((frames_per_clip,)\
                                + obs_shape),
            "right_observation": tf.TensorShape((frames_per_clip,)\
                                + obs_shape),
            "left_tid": tf.TensorShape([]),
            "right_tid": tf.TensorShape([]),
            "left_action": tf.TensorShape((frames_per_clip,) + action_shape),
            "right_action": tf.TensorShape((frames_per_clip,) + action_shape),
            "label": tf.TensorShape([])}
    if answer_type == "ctl":
        output_types["left_moment"] = tf.uint8
        output_types["right_moment"] = tf.uint8
        output_types["ctl_label"] = tf.float32
        
        output_shapes["left_moment"] = tf.TensorShape((moment_len,)\
                                + frame_shape + (1,))
        output_shapes["right_moment"] = tf.TensorShape((moment_len,)\
                                + frame_shape + (1,))
        output_shapes["ctl_label"] = tf.TensorShape([])
    if answer_type == "ctl2":
        output_types["moment"] = tf.uint8
        output_types["ctl_label"] = tf.float32
        output_types["is_ctl_padding"] = tf.float32
        output_shapes["moment"] = tf.TensorShape((moment_len,)\
                                + frame_shape + (1,))
        output_shapes["ctl_label"] = tf.TensorShape([])
        output_shapes["is_ctl_padding"] = tf.TensorShape((moment_len,))
    dataset = tf.data.Dataset.from_generator(
        lambda: load_answer(),
        output_types = output_types,
        output_shapes = output_shapes)
    dataset = dataset.batch(batchsize).prefetch(1)
    element = dataset.make_one_shot_iterator().get_next()
    return element, None

@gin.configurable
def train_reward_model(FLAGS, frames_per_clip, logging_freq):
    tf.reset_default_graph()
    if FLAGS.method == "trex":
        sample, _ = reward_learning_input_fn(answer_type="trex")
    elif FLAGS.method == "crowd_lasso_bt":
        sample, _ = reward_learning_input_fn(answer_type="ctl")
    elif FLAGS.method in ["crowd_lasso_bt2", "crowd_lasso_bt3"]:
        sample, _ = reward_learning_input_fn(answer_type="ctl2")
    else:
        sample, _ = reward_learning_input_fn(answer_type="bt")
    if FLAGS.method == "lasso_bt":
        pref_model = LassoBT("preference_model")
    elif FLAGS.method in ["lasso_bt2", "lasso_bt3"]:
        pref_model = LassoBT("preference_model", warmup_steps=FLAGS.n_steps/2)
    elif FLAGS.method == "crowd_lasso_bt":
        pref_model = CrowdLassoBT(
            "preference_model", warmup_steps=FLAGS.n_steps/2)
    elif FLAGS.method == "crowd_lasso_bt2":
        pref_model = CrowdLassoBT2(
            "preference_model", warmup_steps=FLAGS.n_steps/2)
    elif FLAGS.method == "crowd_lasso_bt3":
        pref_model = CrowdLassoBT3(
            "preference_model", warmup_steps=FLAGS.n_steps/2)
    else:
        pref_model = BT("preference_model")
    bt_loss = pref_model.compute_bt_loss(sample, frames_per_clip)
    step_t = tf.train.get_or_create_global_step()
    if FLAGS.method in ["lasso_bt3", "crowd_lasso_bt", "crowd_lasso_bt2", "crowd_lasso_bt3"]:
        bt_loss += pref_model.compute_reg(sample, frames_per_clip, step_t)
    trainable_vars = pref_model.trainable_variables
    train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                    bt_loss, var_list=trainable_vars,
                    global_step=step_t)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(FLAGS.gpu_id)
    model_path = osp.join(FLAGS.exp_path, FLAGS.game, FLAGS.split,
                            FLAGS.method)
    if not osp.exists(model_path):
        os.mkdir(model_path)
    logging.get_absl_handler().use_absl_log_file('logs', model_path)
    logging.info(trainable_vars)
    ckpt_name = osp.join(model_path, "model")
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    if FLAGS.method == "crowd_lasso_bt3":
        pretrain_path = osp.join(
            FLAGS.exp_path, FLAGS.game, FLAGS.split, "lasso_bt3")
        to_load = [v for v in trainable_vars if not "switch" in v.name]\
                    + [step_t]
        pretrain_loader = tf.train.Saver(save_relative_paths=True,
        var_list = to_load)
    with tf.Session(config=config) as sess:
        if not tf.train.latest_checkpoint(model_path) is None:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            sess.run(tf.global_variables_initializer())
            #if FLAGS.method == "crowd_lasso_bt3":
                #pretrain_loader.restore(sess, tf.train.latest_checkpoint(pretrain_path))
        step = sess.run(step_t)
        while step < FLAGS.n_steps:
            step, _, bt_loss_val = sess.run([step_t, train_step, bt_loss])
            if step % logging_freq == 0:
                print_str = f"step {step}, bt loss={bt_loss_val}."
                logging.info(print_str)
                saver.save(sess, ckpt_name, global_step=step_t)
        saver.save(sess, ckpt_name, global_step=step_t)

@gin.configurable
def compute_clip_reward(FLAGS, preference_path, frames_per_clip, batchsize, frame_shape):
    with bz2.BZ2File(osp.join(preference_path,
                              "trajectories.pbz2"), 'rb') as f:
        trajectories = pickle.load(f)
    def load_test_clips():
        observations = []
        actions = []
        for trajectory in trajectories:
            for _, (clip_start, _) in trajectory["clips"].items():
                obs = trajectory["observation"][clip_start:clip_start+frames_per_clip]
                act = trajectory["action"][clip_start:clip_start+frames_per_clip]
                observations.append(obs)
                actions.append(act)
        if FLAGS.game in atari_observation_masks:
            return  np.array(observations)[:, :, :, :, None], np.array(actions)
        else:
            return np.array(observations), np.array(actions)
    tf.reset_default_graph()
    obs_dtype = trajectories[0]["observation"].dtype
    obs_shape = trajectories[0]["observation"].shape[1:]
    if FLAGS.game in atari_observation_masks:
        obs_shape += (1,)
    action_dtype = trajectories[0]["action"].dtype
    action_shape = trajectories[0]["action"].shape
    if obs_dtype == np.uint8:
        obs_dtype = tf.uint8
    elif obs_dtype == np.float32:
        obs_dtype = tf.float32
    else:
        raise NotImplementedError
    if action_dtype == np.int32:
        action_dtype = tf.int32
    elif action_dtype == np.float32:
        action_dtype = tf.float32
    else:
        raise NotImplementedError
    if len(action_shape) > 1:
        action_shape = action_shape[1:]
    else:
        action_shape = ()
    observations = tf.placeholder(
        obs_dtype, shape=(None, frames_per_clip) + obs_shape)
    actions = tf.placeholder(
        action_dtype, shape=(None, frames_per_clip) + action_shape)
    if FLAGS.method in ["lasso_bt", "lasso_bt2", "lasso_bt3", "crowd_lasso_bt", "crowd_lasso_bt2", "crowd_lasso_bt3"]:
        if FLAGS.method == "crowd_lasso_bt":
            pref_model = CrowdLassoBT("preference_model")
        elif FLAGS.method == "crowd_lasso_bt2":
            pref_model = CrowdLassoBT2("preference_model")
        elif FLAGS.method == "crowd_lasso_bt3":
            pref_model = CrowdLassoBT3("preference_model")
        else:
            pref_model = LassoBT("preference_model")
        rewards_tf = pref_model(observations, actions, frames_per_clip)
        encoded_obs = pref_model.obs_encoder.encode_sequence(observations, frames_per_clip, normalize=False)
        if FLAGS.method == "crowd_lasso_bt3":
            frame_weights = pref_model.clip_encoder(encoded_obs)\
                        * tf.sigmoid(pref_model.switch(encoded_obs))
        else:
            frame_weights = pref_model.clip_encoder(encoded_obs)
        clip_vecs = tf.reduce_sum(encoded_obs*frame_weights, axis=1)
    else:
        pref_model = BT("preference_model")
        rewards_tf = pref_model(observations, actions, frames_per_clip)  
     
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(FLAGS.gpu_id)
    model_path = osp.join(FLAGS.exp_path, FLAGS.game, FLAGS.split,
                            FLAGS.method)
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session(config=config) as sess:
        if not tf.train.latest_checkpoint(model_path) is None:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            sess.run(tf.global_variables_initializer())
        test_clips = load_test_clips()
        if FLAGS.method in ["lasso_bt", "lasso_bt2", "lasso_bt3", "crowd_lasso_bt", "crowd_lasso_bt2", "crowd_lasso_bt3"]:
            rewards = []
            vecs = []
            weights = []
            frames = []
            for bid in range(0, len(test_clips[0]), batchsize):
                reward, vec, weight, frame = sess.run(
                    [rewards_tf, clip_vecs, frame_weights, encoded_obs],
                    feed_dict={observations: test_clips[0][bid: bid + batchsize],
                    actions: test_clips[1][bid: bid + batchsize]})
                rewards.append(reward)
                vecs.append(vec)
                weights.append(weight)
                frames.append(frame)
            rewards = np.concatenate(rewards, axis=0)
            vecs = np.concatenate(vecs, axis=0)
            weights = np.concatenate(weights, axis=0)
            frames = np.concatenate(frames, axis=0)
            np.save(os.path.join(model_path, "clip_reward.npy"), rewards)
            np.save(os.path.join(model_path, "frame_repre.npy"), frames)
            np.save(os.path.join(model_path, "frame_weights.npy"), weights)
            np.save(os.path.join(model_path, "clip_repre.npy"), vecs)
        else:
            rewards = []
            for bid in range(0, len(test_clips[0]), batchsize):
                reward = sess.run(rewards_tf,
                    feed_dict={observations: test_clips[0][bid: bid + batchsize],
                    actions: test_clips[1][bid: bid + batchsize]})
                rewards.append(reward)
            rewards = np.concatenate(rewards, axis=0)
            np.save(os.path.join(model_path, "clip_reward.npy"), rewards)

def inference_input_fn(records, game):
    if game in atari_observation_masks:
        obs_shape = records["observation"].shape[1:]
        def generate_observation_from_log(records):
            mask = atari_observation_masks[game]
            for obs_id, obs in enumerate(records['observation']):
                action = np.array([records['action'][obs_id]])
                obs = obs[None] * mask
                obs = obs[:,:,:,None]
                yield {"observation":obs, "action": action}
        dataset = tf.data.Dataset.from_generator(
            lambda: generate_observation_from_log(records),
            output_types={"observation": tf.uint8,
                        "action": tf.int32},
            output_shapes={"observation": tf.TensorShape((1,) + obs_shape
                                            + (1,)),
                        "action": tf.TensorShape((1,))}
        )
    else:
        obs_shape = records["observations"].shape[1:]
        action_shape = records["actions"].shape
        if len(action_shape) > 1:
            action_shape = action_shape[1:]
        else:
            action_shape = ()
        def generate_observation_from_log(records):
            for obs_id, obs in enumerate(records['observations']):
                action = np.array([records['actions'][obs_id]])
                yield {"observation":obs[None], "action": action}
        dataset = tf.data.Dataset.from_generator(
            lambda: generate_observation_from_log(records),
            output_types={"observation": tf.float32,
                          "action": tf.float32},
            output_shapes={"observation": tf.TensorShape((1,) + obs_shape),
                        "action": tf.TensorShape((1,) + action_shape)}
        )
    return dataset

def compute_reward_for_log(pref_model, records, model_path,
                         batchsize, gpu_id, game):
    tf.reset_default_graph()
    dataset = inference_input_fn(records, game)
    dataset = dataset.batch(batchsize).prefetch(1)
    batch = dataset.make_one_shot_iterator().get_next()
    pref_model = pref_model("preference_model")
    rewards_tf = pref_model(batch['observation'], batch['action'], 1) 
    rewards_tf = tf.reduce_sum(rewards_tf, axis=1)
    rewards_tf = tf.squeeze(rewards_tf, axis=1)
    rewards = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu_id)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        while True:
            try:
                rewards.extend(sess.run(rewards_tf))
            except tf.errors.OutOfRangeError:
                break
    return np.array(rewards)

@gin.configurable
def infer_rewards(FLAGS, log_path,  frame_shape, stack_size, batchsize):
    
    model_path = osp.join(FLAGS.exp_path, FLAGS.game, FLAGS.split,
                            FLAGS.method)
    save_path = osp.join(model_path, "preference_rewards")
    if not osp.exists(save_path):
        os.mkdir(save_path)
    if FLAGS.method in ["lasso_bt", "lasso_bt2", "lasso_bt3"]:
        pref_model = LassoBT
    elif FLAGS.method == "crowd_lasso_bt":
        pref_model = CrowdLassoBT
    elif FLAGS.method == "crowd_lasso_bt2":
        pref_model = CrowdLassoBT2
    elif FLAGS.method == "crowd_lasso_bt3":
        pref_model = CrowdLassoBT3
    else:
        pref_model = BT
    if FLAGS.game in atari_observation_masks:
        log_path = osp.join(log_path, FLAGS.game, FLAGS.split, "replay_logs")
        suffix_list, n_file = enumerate_files(log_path)
        generated_suffix = [f.split(".")[-2] for f in os.listdir(save_path) if f.startswith("$store$")]
        generated_suffix = sorted(generated_suffix)
        generated_suffix = generated_suffix[:-1]
        suffix_list = [f for f in suffix_list if not f in generated_suffix]
        n_file = len(suffix_list)
        logging.info("Generating {} log files.".format(n_file))
        for suffix in suffix_list:
            logging.info("Generating rewards for suffix {}...".format(suffix))
            buffer = DummyWrappedBuffer(observation_shape=frame_shape,
                                        stack_size=stack_size)
            buffer.load(checkpoint_dir=log_path, suffix=suffix)
            records = buffer._store
            rewards = compute_reward_for_log(pref_model, records, model_path, gpu_id=FLAGS.gpu_id, batchsize=batchsize, game=FLAGS.game)
            buffer._store["reward"] = rewards
            buffer.save(save_path, suffix)
            del buffer, rewards
            gc.collect()
    else:
        env = gym.make(FLAGS.game)
        dataset = env.get_dataset()
        rewards = compute_reward_for_log(pref_model, dataset, model_path, gpu_id=FLAGS.gpu_id, batchsize=batchsize, game=FLAGS.game)
        save_path = osp.join(save_path, FLAGS.game + ".hdf5")
        new_dataset = h5py.File(save_path, 'w')
        for k in dataset:
            if k == "rewards":
                entry = rewards
            elif k.startswith("metadata"):
                continue
            else:
                entry = dataset[k]
            new_dataset.create_dataset(k, data=entry, compression="gzip")
        new_dataset.close()

@gin.configurable
def generate_perturbed_expert_demo_datasets(FLAGS, atari_log_path=None, ratio=0.1, perturb_type="reward"):
    model_path = osp.join(FLAGS.exp_path, FLAGS.game, FLAGS.split,
                            FLAGS.method)
    save_path = osp.join(
        FLAGS.exp_path+f"_{perturb_type}_perturbed_{int(100*ratio)}",
        FLAGS.game, FLAGS.split, FLAGS.method, "preference_rewards")
    os.makedirs(save_path, exist_ok=True)
    if FLAGS.method in ["lasso_bt", "lasso_bt2", "lasso_bt3", "crowd_lasso_bt"]:
        pref_model = LassoBT
    elif FLAGS.method == "crowd_lasso_bt2":
        pref_model = CrowdLassoBT2
    elif FLAGS.method == "crowd_lasso_bt3":
        pref_model = CrowdLassoBT3
    else:
        pref_model = BT
    if FLAGS.game in atari_observation_masks:
        log_path = osp.join(atari_log_path, FLAGS.game, FLAGS.split, "replay_logs")
        suffix_list = sorted([int(s) for s in enumerate_files(log_path)[0]])
        n_file = len(suffix_list)
        logging.info(f"Generating {n_file} log files.")
        scores, rewards = [], []
        for suffix in suffix_list:
            logging.info(f"Computing scores for suffix {suffix}...")
            if perturb_type == "random":
                score = np.random.random(1000000)
            else:
                buffer = DummyWrappedBuffer(observation_shape=(84,84),
                                        stack_size=4)
                buffer.load(checkpoint_dir=log_path, suffix=suffix)
                records = buffer._store
                if perturb_type == "reward":
                    reward = compute_reward_for_log(
                    pref_model, records, model_path, gpu_id=FLAGS.gpu_id,
                    batchsize=32, game=FLAGS.game)
                    score = np.abs(reward)
                elif perturb_type == "weight":
                    score = np.abs(compute_weights_for_log(
                        pref_model, records, model_path, gpu_id=FLAGS.gpu_id,
                        batchsize=32, game=FLAGS.game))
                else:
                    raise NotImplementedError
                if ratio == 0:
                    buffer.save(save_path, suffix, ["terminal"])
                else:
                    scores.append(score)
                del buffer
                gc.collect()
            scores.append(score)
        if ratio > 0:
            threshold_ratio = 1 - ratio
            threshold_value = np.quantile(
                np.concatenate(scores, axis=0), threshold_ratio)
            logging.info(f"Cutting off threshold: {threshold_value}")
            for sid, suffix in enumerate(suffix_list):
                logging.info(f"Perturbating suffix {suffix}...")
                buffer = DummyWrappedBuffer(observation_shape=(84,84),
                                        stack_size=4)
                buffer.load(checkpoint_dir=log_path, suffix=suffix)
                records = buffer._store
                terminals = buffer._store["terminal"]
                for i in range(1, 1000000):
                    if scores[sid][i] >= threshold_value:
                        terminals[i] = 1.
                        terminals[i-1] = 1.
                buffer._store["terminal"] = terminals
                buffer.save(save_path, suffix, ["terminal"])
                del buffer
                gc.collect()
    else:
        env = gym.make(FLAGS.game)
        dataset = env.get_dataset()
        rewards = compute_reward_for_log(pref_model, dataset, model_path, gpu_id=FLAGS.gpu_id, batchsize=32, game=FLAGS.game)
        if perturb_type == "reward":
            score = np.abs(rewards)
        elif perturb_type == "weight":
            score = np.abs(compute_weights_for_log(
                pref_model, dataset, model_path, gpu_id=FLAGS.gpu_id,
                batchsize=32, game=FLAGS.game))
        elif perturb_type == "random":
            score = np.random.random(len(rewards))
        else:
            raise NotImplementedError
        threshold_ratio = 1 - ratio
        threshold_value = np.quantile(score, threshold_ratio) 
        should_keep = []
        for i in range(len(score)):
            if score[i] >= threshold_value:
                should_keep.append(0)
                if i == 0:
                    continue
                dataset['terminals'][i-1] = True
            else:
                should_keep.append(1)
        should_keep = np.array(should_keep)
        should_keep = should_keep > 0
        save_path = osp.join(save_path, FLAGS.game + ".hdf5")
        new_dataset = h5py.File(save_path, 'w')
        for k in dataset:
            if k == "rewards":
                entry = rewards
            elif k.startswith("metadata"):
                continue
            else:
                entry = dataset[k]
            new_dataset.create_dataset(k, data=entry[should_keep], compression="gzip")
        new_dataset.close()
    
def compute_weights_for_log(pref_model, records, model_path,
                         batchsize, gpu_id, game):
    tf.reset_default_graph()
    dataset = inference_input_fn(records, game)
    dataset = dataset.batch(batchsize).prefetch(1)
    batch = dataset.make_one_shot_iterator().get_next()
    pref_model = pref_model("preference_model")
    rewards_tf = pref_model(batch['observation'], batch['action'], 1)
    weights_tf = pref_model.compute_weights(batch['observation'], batch['action'], 1) 
    weights_tf = tf.reduce_sum(weights_tf, axis=1)
    weights_tf = tf.squeeze(weights_tf, axis=1)
    weights = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu_id)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        while True:
            try:
                weights.extend(sess.run(weights_tf))
            except tf.errors.OutOfRangeError:
                break
    return np.array(weights)