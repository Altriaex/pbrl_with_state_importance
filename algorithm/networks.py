'''
Descripttion: 
Author: CoeusZhang
Date: 2022-02-14 11:41:57
LastEditTime: 2022-03-20 12:19:49
'''
import gin
import tensorflow.compat.v1 as tf
import numpy as np

@gin.configurable
class ObservationEncoder(tf.keras.Model):
    def __init__(self, name="obs_encoder", fc_dim=512, emb_dim=128):
        super(ObservationEncoder, self).__init__(name=name)
        activation_fn = tf.keras.activations.elu
        self.conv1 = tf.keras.layers.Conv2D(
            16, [8, 8], strides=4, padding='same',
            activation=activation_fn, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            16, [4, 4], strides=2, padding='same',
            activation=activation_fn, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(
            16, [3, 3], strides=1, padding='same',
            activation=activation_fn, name='conv3')
        # batchsize, frames_per_clip, 11, 11, 16
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            fc_dim, activation=activation_fn, name='dense1')
        self.dense2 = tf.keras.layers.Dense(emb_dim, name='dense2')
        self.emb_dim = emb_dim

    def call(self, obs, normalize=False):
        obs = tf.cast(obs, tf.float32) / 255
        for layer in [self.conv1, self.conv2, self.conv3,
                      self.flatten, self.dense1, self.dense2]:
            obs = layer(obs)
        if normalize:
            obs = tf.math.l2_normalize(obs, axis=1)
        return obs
    
    def encode_sequence(self, obs_seq, n_frames, normalize=False):
        obs_seq = tf.cast(obs_seq, tf.float32) / 255
        encoded = []
        for fid in range(n_frames):
            frame = obs_seq[:, fid]
            for layer in [self.conv1, self.conv2, self.conv3,
                          self.flatten, self.dense1, self.dense2]:
                frame = layer(frame)
            if normalize:
                frame = tf.math.l2_normalize(frame, axis=1)
            encoded.append(frame[:, None])
        encoded = tf.concat(encoded, axis=1)
        return encoded

@gin.configurable
class ContinuousObservationEncoder(tf.keras.Model):
    def __init__(self, name="obs_encoder", fc_dim=256, emb_dim=128):
        super(ContinuousObservationEncoder, self).__init__(name=name)
        activation_fn = tf.keras.activations.elu
        self.dense1 = tf.keras.layers.Dense(
            fc_dim, activation=activation_fn, name='dense1')
        self.dense2 = tf.keras.layers.Dense(
            int(fc_dim/2), activation=activation_fn, name='dense1')
        self.dense3 = tf.keras.layers.Dense(emb_dim, name='dense3')
        self.emb_dim = emb_dim

    def call(self, obs, normalize=False):
        for layer in [self.dense1, self.dense2, self.dense3]:
            obs = layer(obs)
        if normalize:
            obs = tf.math.l2_normalize(obs, axis=1)
        return obs
    
    def encode_sequence(self, obs_seq, n_frames, normalize=False):
        encoded = []
        for fid in range(n_frames):
            frame = obs_seq[:, fid]
            for layer in [self.dense1, self.dense2, self.dense3]:
                frame = layer(frame)
            if normalize:
                frame = tf.math.l2_normalize(frame, axis=1)
            encoded.append(frame[:, None])
        encoded = tf.concat(encoded, axis=1)
        return encoded

@gin.configurable
class ClipEncoder(tf.keras.Model):
    def __init__(self, name, frames_per_clip, emb_dim, l1_weight, temporal_l2_weight, warmup_steps):
        super(ClipEncoder, self).__init__(name=name)
        activation_fn = tf.keras.activations.elu
        self.dense1 = tf.keras.layers.Dense(
            emb_dim, activation=activation_fn, name='dense1')
        self.dense2 = tf.keras.layers.Dense(
            1,  name='dense2')
        self.frames_per_clip = frames_per_clip
        self.emb_dim = emb_dim
        self.l1_weight = l1_weight
        self.temporal_l2_weight = temporal_l2_weight
        self.warmup_steps = float(warmup_steps)

    def call(self, encoded_obs):
        weights = self.dense2(self.dense1(encoded_obs))
        return weights
    
    def get_temporal_l2_weighting_matrix(self):
        weighting = np.zeros((self.frames_per_clip-1, self.frames_per_clip),
                             dtype=np.float32)
        for i in range(self.frames_per_clip-1):
            weighting[i,i] = 1
            weighting[i,i+1] = -1
        return tf.convert_to_tensor(weighting)
    
    def compute_weighting_reg(self, weights, step_t):
        #batch, frames_per_clip - 1
        reg = self.compute_temporal_l2_reg(weights, step_t)
        weights = tf.squeeze(weights, axis=2)
        l1_reg = tf.reduce_mean(tf.reduce_sum(tf.abs(weights), axis=1))
        l1_reg *= self.l1_weight
        reg += l1_reg * tf.minimum(
                    tf.cast(step_t, tf.float32)/ self.warmup_steps, 1.0)
        
        return reg
    
    def compute_temporal_l2_reg(self, weights, step_t):
        weights = tf.squeeze(weights, axis=2)
        temporal_l2_weighting = self.get_temporal_l2_weighting_matrix()
        temporal_l2_reg = tf.matmul(
            weights, temporal_l2_weighting, transpose_b=True)
        temporal_l2_reg = tf.reduce_sum(temporal_l2_reg**2, axis=1)
        temporal_l2_reg = tf.reduce_mean(temporal_l2_reg)
        temporal_l2_reg *= self.temporal_l2_weight
        reg = temporal_l2_reg* tf.minimum(
                    tf.cast(step_t, tf.float32)/ self.warmup_steps, 1.0)
        return reg


class FC(tf.keras.Model):
    def __init__(self, name):
        super(FC, self).__init__(name=name)
        activation_fn = tf.keras.activations.elu
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn, name='dense1')
        self.dense2 = tf.keras.layers.Dense(
            1,  name='dense2')

    def call(self, encoded_obs):
        return self.dense2(self.dense1(encoded_obs))
