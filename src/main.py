import learnrl as rl
import tensorflow as tf
import retro

from agents.a2c import ActorCriticAgent

env = retro.make(game='Airstriker-Genesis')
agent = ActorCriticAgent(env.action_space)

print(env.observation_space, env.action_space)

reward_handler = rl.RewardHandler

pg = rl.Playground(env, agent)
pg.fit(100, render=True, verbose=2, titles_on_top=False, )
pg.test(1, titles_on_top=False)
