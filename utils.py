"""
Descripttion: 
version: 
Author: Altriaex
Date: 2021-05-07 20:54:49
LastEditors: Altriaex
LastEditTime: 2021-05-08 00:43:00
"""
'''
Description: 
Author: CoeusZhang
Date: 2021-05-03 18:04:22
LastEditTime: 2022-03-26 16:43:36
'''
import ffmpeg
import numpy as np
from dopamine.discrete_domains import atari_lib
import os.path as osp
import os
from .dummy_wrapped_buffer import DummyWrappedBuffer
import zipfile
FRAME_PER_CLIP = 180
FRAME_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
BATCH_SIZE = 24

def frames2video(file_name, images, framerate=60, vcodec='libx264', pix_fmt ='gray',
        scale=1):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, 
                    s='{}x{}'.format(width, height))
            .output(file_name, pix_fmt='yuv420p', vcodec=vcodec, r=framerate,
                    vf="scale={}:-1".format(scale*width))
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=False)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

def load_logs(logs_dir, suffix):
    buffer = DummyWrappedBuffer(
    observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
    stack_size=atari_lib.NATURE_DQN_STACK_SIZE)
    buffer.load(checkpoint_dir=logs_dir, suffix=suffix)
    # A dict with keys: observation, action, reward, terminal
    return buffer._store

atari_observation_masks = {
    "Seaquest": (np.array([[0.] * 12 + [1.] * 72] * 84, dtype=np.uint8).T)[None],
    "Pong": (np.array([[0.] * 12 + [1.] * 72] * 84, dtype=np.uint8).T)[None],
    "BeamRider": (np.array([[0.] * 12 + [1.] * 72] * 84, dtype=np.uint8).T)[None],
    "Enduro": (np.array([[1.] * 64 + [0.] * 20] * 84, dtype=np.uint8).T)[None],
    "Breakout": (np.array([[0.] * 6 + [1.] * 78] * 84, dtype=np.uint8).T)[None],
    "Qbert": (np.array([[0.] * 12 + [1.] * 72] * 84, dtype=np.uint8).T)[None]}
atari_observation_masks["Alien"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["Alien"][70:77, 10:37] = 0
atari_observation_masks["Alien"] = atari_observation_masks["Alien"][None]
atari_observation_masks["Hero"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["Hero"][70:77, 30:55] = 0
atari_observation_masks["Hero"] = atari_observation_masks["Hero"][None]
atari_observation_masks["PrivateEye"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["PrivateEye"][:7, :] = 0
atari_observation_masks["PrivateEye"] = atari_observation_masks["PrivateEye"][None]
atari_observation_masks["Asterix"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["Asterix"][3:7, 35:54] = 0
atari_observation_masks["Asterix"] = atari_observation_masks["Asterix"][None]
atari_observation_masks["Boxing"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["Boxing"][0:5, 20:62] = 0
atari_observation_masks["Boxing"] = atari_observation_masks["Boxing"][None]
atari_observation_masks["Assault"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["Assault"][13:16, 29:54] = 0
atari_observation_masks["Assault"][76:80, 0:40] = 0
atari_observation_masks["Assault"] = atari_observation_masks["Assault"][None]
atari_observation_masks["Amidar"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["Amidar"][65:, 35:] = 0
atari_observation_masks["Amidar"] = atari_observation_masks["Amidar"][None]
atari_observation_masks["BattleZone"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["BattleZone"][71:, 34:] = 0
atari_observation_masks["BattleZone"] = atari_observation_masks["BattleZone"][None]
atari_observation_masks["RoadRunner"] = np.ones((84,84), dtype=np.uint8)
atari_observation_masks["RoadRunner"][:18, 29:55] = 0
atari_observation_masks["RoadRunner"] = atari_observation_masks["RoadRunner"][None]