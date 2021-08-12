"""
Copyright (C) 2020, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import collections

import numpy as np
import gym


class Vector(gym.ObservationWrapper):
    def __init__(self, env, stack=4):
        super().__init__(env)
        self.stack = stack
        self.game = self.env._game
        self.arena = self.game._arena
        self.observation_space = gym.spaces.Box(0, 1, [2 + stack * 2])
        self.history = collections.deque([self.get_position_ball()] * stack, maxlen=stack)
        env._get_screen_img_double_player = lambda: [None, None]

    def get_position_ball(self):
        width = self.arena.right - self.arena.left
        height = self.arena.bottom - self.arena.top
        ball = self.game._ball
        return np.array([(ball._rect.x - self.arena.left) / width, (ball._rect.y - self.arena.top) / height])

    def get_position_bat_left(self):
        height = self.arena.bottom - self.arena.top
        return (self.game._left_bat._rect.y - self.arena.top) / height

    def get_position_bat_right(self):
        height = self.arena.bottom - self.arena.top
        return (self.game._right_bat._rect.y - self.arena.top) / height

    def reset(self, *args, **kwargs):
        return self.env._reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.env._step(*args, **kwargs)

    def observation(self, observation):
        position = self.get_position_ball()
        self.history.append(position)
        return np.concatenate([[self.get_position_bat_left(), self.get_position_bat_right()], np.concatenate(self.history)])
