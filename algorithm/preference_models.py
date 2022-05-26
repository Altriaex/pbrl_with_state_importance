'''
Descripttion: 
Author: CoeusZhang
Date: 2022-03-14 09:15:10
LastEditTime: 2022-05-26 17:43:18
'''
import gin
import tensorflow.compat.v1 as tf
from .networks import ObservationEncoder, ClipEncoder, ContinuousObservationEncoder

@gin.configurable
class BT(tf.keras.Model):
    def __init__(self, name, n_action, obs_encoder):
        super(BT, self).__init__(name=name)
        self.obs_encoder = obs_encoder(name="obs_encoder")
        self.reward_network = tf.keras.layers.Dense(
            1, name="reward_network")
    
    def call(self, obs, action, frames_per_clip):
        encoded_obs = self.obs_encoder.encode_sequence(
            obs, frames_per_clip, normalize=False)
        encoded_obs = tf.keras.activations.elu(encoded_obs)
        all_reward = self.reward_network(encoded_obs)
        #reward = tf.gather(all_reward, action[:, :, None], batch_dims=2, axis=2)
        return all_reward

    @property
    def vars_to_load(self):
        vars_to_load = self.obs_encoder.trainable_variables
        assign_map = {}
        for v in vars_to_load:
            name = v.name.split("/")
            name_in_ckpt = "/".join(name[1:])[:-2]
            assign_map[name_in_ckpt] = v
        return assign_map
    
    @property
    def trainable_variables(self):
        return self.reward_network.trainable_variables + self.obs_encoder.trainable_variables
    
    def compute_bt_loss(self, sample, frames_per_clip):
        observations = tf.concat(
            [sample["left_observation"], sample["right_observation"]], axis=0)
        actions = tf.concat(
            [sample["left_action"], sample["right_action"]], axis=0)
        sample_in_batchsize = tf.shape(sample["left_observation"])[0]
        rewards = self(observations, actions, frames_per_clip)
        sum_rewards = tf.reduce_sum(rewards, axis=1)
        l_sum_rewards = sum_rewards[:sample_in_batchsize]
        r_sum_rewards = sum_rewards[sample_in_batchsize:]
        bt_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=sample["label"][:, None], logits=l_sum_rewards - r_sum_rewards))
        return bt_loss

@gin.configurable
class LassoBT(BT):
    def __init__(self, name, n_action, obs_encoder, warmup_steps=1):
        super(LassoBT, self).__init__(name=name, n_action=n_action, obs_encoder=obs_encoder)
        self.clip_encoder = ClipEncoder(
            name="clip_encoder", warmup_steps=warmup_steps)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.reward_vec = tf.get_variable("reward_vec",
            (self.obs_encoder.emb_dim,))
    
    def call(self, obs, action, frames_per_clip):
        encoded_obs = self.obs_encoder.encode_sequence(
            obs, frames_per_clip, normalize=False)
        frame_weights = self.clip_encoder(encoded_obs)
        encoded_obs = tf.reduce_sum(encoded_obs * frame_weights, axis=1)
        # encoded_obs: batchsize, obs_dim
        reward = tf.reduce_sum(encoded_obs * self.reward_vec[None,], axis=1)
        return reward[:, None, None]
    
    def compute_weights(self, obs, action, frames_per_clip):
        encoded_obs = self.obs_encoder.encode_sequence(
            obs, frames_per_clip, normalize=False)
        frame_weights = self.clip_encoder(encoded_obs)
        return frame_weights

    def compute_reg(self, sample, frames_per_clips, step_t):
        observations = tf.concat(
            [sample["left_observation"], sample["right_observation"]], axis=0)
        encoded_obs = self.obs_encoder.encode_sequence(
            observations, frames_per_clips)
        frame_weights = self.clip_encoder(encoded_obs)
        reg = self.clip_encoder.compute_weighting_reg(frame_weights, step_t)
        return reg

    @property
    def trainable_variables(self):
        return [self.reward_vec] + self.clip_encoder.trainable_variables + self.obs_encoder.trainable_variables
