"""
Preprocessing utilities for Atari environments.

This module provides wrapper classes for preprocessing Atari game frames:
- Grayscale conversion
- Frame resizing
- Frame stacking
- Frame skipping
- Reward clipping
"""

import gymnasium as gym
import numpy as np
from collections import deque
import cv2


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, _ = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame (frameskipping) and max pool over the 
    last 2 frames to handle flickering sprites in Atari games.
    """
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Max pool over the last 2 frames
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {-1, 0, 1} by their sign.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper and later work.
    Converts to grayscale.
    """
    def __init__(self, env, width=84, height=84, grayscale=True):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 1),
                dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8
            )

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class FrameStack(gym.Wrapper):
    """
    Stack k last frames. Returns lazy array, which is memory efficient.
    """
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=2)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Normalize pixel values to [0, 1].
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def make_atari_env(env_id, frame_stack=4, clip_rewards=True, noop_max=30):
    """
    Create a wrapped Atari environment with all standard preprocessing.
    
    Args:
        env_id: Gymnasium environment ID (e.g., 'ALE/SpaceInvaders-v5')
        frame_stack: Number of frames to stack (default: 4)
        clip_rewards: Whether to clip rewards to {-1, 0, 1} (default: True)
        noop_max: Maximum number of no-op actions on reset (default: 30)
    
    Returns:
        Wrapped Gymnasium environment
    """
    env = gym.make(env_id, render_mode='rgb_array')
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=frame_stack)
    env = ScaledFloatFrame(env)
    return env


def make_atari_env_rgb(env_id, frame_stack=4, clip_rewards=True, noop_max=30):
    """
    Create a wrapped Atari environment with RGB frames (for visualization).
    
    Args:
        env_id: Gymnasium environment ID (e.g., 'ALE/SpaceInvaders-v5')
        frame_stack: Number of frames to stack (default: 4)
        clip_rewards: Whether to clip rewards to {-1, 0, 1} (default: True)
        noop_max: Maximum number of no-op actions on reset (default: 30)
    
    Returns:
        Wrapped Gymnasium environment with RGB frames
    """
    env = gym.make(env_id, render_mode='rgb_array')
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=84, height=84, grayscale=False)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=frame_stack)
    return env
