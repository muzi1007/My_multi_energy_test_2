import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import sys
sys.path.append("..")
from env_v2 import *

def scale_actions(action, min_val, max_val):
    """
    將原始動作 (範圍為 -1 到 1) 縮放到指定的最小值和最大值。

    Parameters:
    - action: 原始動作，範圍為 -1 到 1。
    - min_val (float): 指定的最小值。
    - max_val (float): 指定的最大值。

    Returns:
    - scaled_action: 縮放後的動作，範圍為 min_val 到 max_val。
    """
    scaled_action = 0.5 * (action + 1.0) * (max_val - min_val) + min_val
    return scaled_action


if __name__ == '__main__':
    env = env_v2()
    agent = Agent(input_dims=[env.state_space], env=env, n_actions=env.action_space)

    n_games = 3000
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'env_v2-v1.png'
    figure_file = 'plots/' + filename

    best_score = -float('inf')
    score_history = []
    load_checkpoint = False

    #n_steps = 0

    if load_checkpoint:
        agent.load_models()
        #env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        n_steps = 0
        while not done:
            actions = agent.choose_action(observation)
            #print('pre_actions: ', actions)
            # actions[0] = scale_actions(actions[0], -1 * observation[5] / 100 * env.electric_net.E_Bat1_E, (100 - observation[5]) / 100 * env.electric_net.E_Bat1_E) #
            actions[0] = scale_actions(actions[0], max(-1 * observation[5] / 100 * env.electric_net.E_Bat1_E, -2), min((100 - observation[5]) / 100 * env.electric_net.E_Bat1_E, 2))
            actions[1] = scale_actions(actions[1], 0, 20) # CHP_p
            actions[2] = scale_actions(actions[2], 0, 20) # Heat_Pump_m
            actions[3] = scale_actions(actions[3], 0, 20) # Natural_Gas_Boiler_m
            actions[4] = scale_actions(actions[4], -1 * observation[7] / 100 * env.thermal_net.Th_Bat1_E, (100 - observation[7]) / 100 * env.thermal_net.Th_Bat1_E)/60/60 # mass_storage_m
            #print('scaled_actions: ', actions)
            #print('observation: ', observation)
            #print('n_steps: ', n_steps)
            observation_, reward, done = env.step(n_steps, actions)
            #print('reward: ', reward)
            n_steps += 1
            score += reward
            agent.remember(observation, actions, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            if n_steps == 1000:
                done = True
            #print('---------------------------------------------------------------------------------------------')

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            '''
            if not load_checkpoint:
                agent.save_models()
            '''
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'time_steps', n_steps)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

