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

from pong_vec import pong_env, wrapper


def test():
    env = pong_env.PongSinglePlayerEnv()
    env = wrapper.Vector(env)
    state = env.reset()
    while True:
        state, reward, done, info = env.step(0)
        if done:
            break
