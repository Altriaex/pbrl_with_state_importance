'''
Description: 
Author: CoeusZhang
Date: 2021-05-08 12:12:43
LastEditTime: 2022-05-16 10:44:25
'''
import os
import os.path as osp
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import gin
from absl import app, flags
from . import trajectory_representation_learning, utils
import zipfile

flags.DEFINE_enum('task', None, ['sample_trajectories', 'train_encoder', 'compute_pretrain_repre', 'sample_queries', 'generate_bt_answers', 'generate_trex_answers', 'train_reward_model', 'compute_clip_reward', 'infer_rewards', 'train_policy', 'test_policy', 'sample_ctl_queries','sample_ctl_queries2', 'generate_perturbation_datasets', 'sample_expert_trajectories'], 
    'The task to execute.')
flags.DEFINE_string('exp_path', None, 'path to root directory for experiments.')
flags.DEFINE_string('game', 'Pong', 'the game to run experiments.')
flags.DEFINE_string('gpu_id', '0', 'the gpu to use.')
flags.DEFINE_string('split', '1', 'The split of data to use')
flags.DEFINE_enum('method', 'bt', ['bt', 'lasso_bt', 'trex', "lasso_bt2", "lasso_bt3", "crowd_lasso_bt", "crowd_lasso_bt2", "crowd_lasso_bt3"], 'The reward learning method')
flags.DEFINE_float('lr', 1e-4, 'The learning rate')
flags.DEFINE_integer('n_steps', 10000, '#steps to train')
flags.DEFINE_float('l1_weight', 1e-2, 'The weight for l1 reg')
flags.DEFINE_float('perturb_ratio', 0.1, 'The ratio of perturbation')
flags.DEFINE_string('perturb_type', 'reward', 'The type of perturbation')
flags.DEFINE_float('temporal_l2_weight', 1e-2, 'The weight for temporal l2 reg')
FLAGS = flags.FLAGS

def pack_preference_rewards(FLAGS):
    exp_base, exp_id = osp.split(FLAGS.exp_path)
    reward_dst = osp.join(exp_base, "preference_rewards")
    if not osp.exists(reward_dst):
        os.mkdir(reward_dst)
    filename = "_".join(
        [exp_id, FLAGS.game, FLAGS.split, FLAGS.method])
    zf = zipfile.ZipFile(osp.join(reward_dst, filename+".zip"), "w")
    reward_path = osp.join(FLAGS.exp_path, FLAGS.game, FLAGS.split,
                           FLAGS.method)
    reward_path = osp.join(reward_path, "preference_rewards")
    for f in os.listdir(reward_path):
        zf.write(osp.join(reward_path, f), arcname=f)
    zf.close()

def pack_perturbed_dataset(FLAGS):
    exp_base, exp_id = osp.split(FLAGS.exp_path)
    reward_dst = osp.join(exp_base, "preference_rewards")
    os.makedirs(reward_dst, exist_ok=True)
    filename = "_".join(
        [exp_id+f"_{FLAGS.perturb_type}_perturbed_{int(100*FLAGS.perturb_ratio)}",
            FLAGS.game, FLAGS.split, FLAGS.method])
    zf = zipfile.ZipFile(osp.join(reward_dst, filename+".zip"), "w")
    reward_path = osp.join(
        FLAGS.exp_path+f"_{FLAGS.perturb_type}_perturbed_{int(100*FLAGS.perturb_ratio)}",
        FLAGS.game, FLAGS.split, FLAGS.method, "preference_rewards")
    for f in os.listdir(reward_path):
        zf.write(osp.join(reward_path, f), arcname=f)
    zf.close()

def main(argv):
    gin.parse_config_file(osp.join(FLAGS.exp_path, 'config.gin'))
    trajectory_representation_learning.set_preference_path(FLAGS)
    trajectory_representation_learning.set_action_space(FLAGS)
    if FLAGS.game in utils.atari_observation_masks:
        for key in ['BT.obs_encoder', 'LassoBT.obs_encoder', 'CrowdLassoBT.obs_encoder', 'CrowdLassoBT2.obs_encoder', 'CrowdLassoBT3.obs_encoder']:
            gin.bind_parameter(key, trajectory_representation_learning.ObservationEncoder)
    else:
        for key in ['BT.obs_encoder', 'LassoBT.obs_encoder', 'CrowdLassoBT.obs_encoder', 'CrowdLassoBT2.obs_encoder', 'CrowdLassoBT3.obs_encoder']:
            gin.bind_parameter(key, trajectory_representation_learning.ContinuousObservationEncoder)
    if FLAGS.task == "sample_trajectories":
        if FLAGS.game in utils.atari_observation_masks:
            trajectory_representation_learning.sample_trajectories(log_split=FLAGS.split)
        else:
            trajectory_representation_learning.sample_d4rl_trajectories(game=FLAGS.game)
    elif FLAGS.task == "train_encoder":
        trajectory_representation_learning.train_encoder(FLAGS)
    elif FLAGS.task == "compute_pretrain_repre":
        trajectory_representation_learning.compute_pretrain_repre(FLAGS)
    elif FLAGS.task == "sample_queries":
        trajectory_representation_learning.sample_queries()
    elif FLAGS.task == "generate_bt_answers":
        trajectory_representation_learning.generate_bt_answers()
    elif FLAGS.task == "generate_trex_answers":
        trajectory_representation_learning.generate_trex_answers()
    elif FLAGS.task == "train_reward_model":
        gin.bind_parameter('ClipEncoder.l1_weight', FLAGS.l1_weight)
        gin.bind_parameter('ClipEncoder.temporal_l2_weight', FLAGS.temporal_l2_weight)
        trajectory_representation_learning.train_reward_model(FLAGS)
    elif FLAGS.task == "compute_clip_reward":
        trajectory_representation_learning.compute_clip_reward(FLAGS)
    elif FLAGS.task == "infer_rewards":
        trajectory_representation_learning.infer_rewards(FLAGS)
        pack_preference_rewards(FLAGS)
    elif FLAGS.task == "train_policy":
        trajectory_representation_learning.train_policy(FLAGS)
    elif FLAGS.task == "test_policy":
        trajectory_representation_learning.test_policy(FLAGS)
    elif FLAGS.task == "sample_ctl_queries":
        trajectory_representation_learning.sample_ctl_queries(FLAGS)
    elif FLAGS.task == "sample_ctl_queries2":
        trajectory_representation_learning.sample_ctl_queries2(FLAGS)
    elif FLAGS.task == "generate_perturbation_datasets":
        trajectory_representation_learning.generate_perturbed_expert_demo_datasets(FLAGS, ratio=FLAGS.perturb_ratio, perturb_type=FLAGS.perturb_type)
        pack_perturbed_dataset(FLAGS)
    elif FLAGS.task == "sample_expert_trajectories":
        if FLAGS.game in utils.atari_observation_masks:
            trajectory_representation_learning.sample_trajectories(log_split=FLAGS.split)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        
if __name__ == '__main__':
    flags.mark_flag_as_required('task')
    flags.mark_flag_as_required('exp_path')
    app.run(main)
