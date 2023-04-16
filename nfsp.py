import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import random
import numpy as np
from collections import deque

from common.utils import epsilon_scheduler, update_target
from model import DQN, Policy
from storage import ReplayBuffer, ReservoirBuffer

from tqdm.notebook import tqdm

device = 'cpu'
eps_start = 1
eps_final = 0.05
eps_decay = 420000#30000
lr = 0.00005
gamma = 0.8
eta = 0.1
multi_step = 3000
buffer_size = 5000
max_frames = 1000000#1500000
n_update_target = 1000
rl_start = 1000
sl_start = 1000
train_freq = 10
evaluation_interval = 1000
batch_size = 64
from automation import automation,battle,random_battle,random_battle_env,general_battle_env

def pk_ai_sim(model,n_round = 1000):
    win = 0
    n = 0
    ratio = []
    for i in range(n_round):
        env = general_battle_env(print_info=False)
        while not env.end:
            a_t = model.act(torch.FloatTensor(env.get_status()[1]))
            action = random.choice([0,1,2])
            env.step(action_a=int(action),action_b=int(a_t))
        n+=1
        win+=(1-env.result)
        ratio.append(win/n)
    return ratio



def train(env):
    # RL Model for Player 1  
    p1_ratio = []
    p2_ratio = []
    p1_current_model = DQN(env).to(device)
    p1_target_model = DQN(env).to(device)
    update_target(p1_current_model, p1_target_model)

    # RL Model for Player 2
    p2_current_model = DQN(env).to(device)
    p2_target_model = DQN(env).to(device)
    update_target(p2_current_model, p2_target_model)

    # SL Model for Player 1, 2
    p1_policy = Policy(env).to(device)
    p2_policy = Policy(env).to(device)

    # if args.load_model and os.path.isfile(args.load_model):
    #     load_model(models={"p1": p1_current_model, "p2": p2_current_model},
    #                policies={"p1": p1_policy, "p2": p2_policy}, args=args)

    epsilon_by_frame = epsilon_scheduler(eps_start, eps_final, eps_decay)

    # Replay Buffer for Reinforcement Learning - Best Response
    p1_replay_buffer = ReplayBuffer(buffer_size)
    p2_replay_buffer = ReplayBuffer(buffer_size)

    # Reservoir Buffer for Supervised Learning - Average Strategy
    # TODO(Aiden): How to set buffer size of SL?
    p1_reservoir_buffer = ReservoirBuffer(buffer_size)
    p2_reservoir_buffer = ReservoirBuffer(buffer_size)

    # Deque data structure for multi-step learning
    p1_state_deque = deque(maxlen=multi_step)
    p1_reward_deque = deque(maxlen=multi_step)
    p1_action_deque = deque(maxlen=multi_step)

    p2_state_deque = deque(maxlen=multi_step)
    p2_reward_deque = deque(maxlen=multi_step)
    p2_action_deque = deque(maxlen=multi_step)

    # RL Optimizer for Player 1, 2
    p1_rl_optimizer = optim.Adam(p1_current_model.parameters(), lr=lr)
    p2_rl_optimizer = optim.Adam(p2_current_model.parameters(), lr=lr)

    # SL Optimizer for Player 1, 2
    # TODO(Aiden): Is it necessary to seperate learning rate for RL/SL?
    p1_sl_optimizer = optim.Adam(p1_policy.parameters(), lr=lr)
    p2_sl_optimizer = optim.Adam(p2_policy.parameters(), lr=lr)

    # Logging
    length_list = []
    p1_reward_list, p1_rl_loss_list, p1_sl_loss_list = [], [], []
    p2_reward_list, p2_rl_loss_list, p2_sl_loss_list = [], [], []
    p1_episode_reward, p2_episode_reward = 0, 0
    tag_interval_length = 0

    # Main Loop
    (p1_state, p2_state) = env.get_status()
    for frame_idx in tqdm(range(1, max_frames + 1)):
        is_best_response = False
        # Action should be decided by a combination of Best Response and Average Strategy
        if random.random() > eta:
            p1_action = p1_policy.act(torch.FloatTensor(p1_state).to(device))
            p2_action = p2_policy.act(torch.FloatTensor(p1_state).to(device))
        else:
            is_best_response = True
            epsilon = epsilon_by_frame(frame_idx)
            p1_action = p1_current_model.act(torch.FloatTensor(p1_state).to(device), epsilon)
            p2_action = p2_current_model.act(torch.FloatTensor(p2_state).to(device), epsilon)

        env.step(p1_action,p2_action)
        (p1_next_state, p2_next_state) = env.get_status()
        reward = env.reward
        done = env.end

        # Save current state, reward, action to deque for multi-step learning
        p1_state_deque.append(p1_state)
        p2_state_deque.append(p2_state)
        
        p1_reward = reward[0]
        p2_reward = reward[1]
        p1_reward_deque.append(p1_reward)
        p2_reward_deque.append(p2_reward)

        p1_action_deque.append(p1_action)
        p2_action_deque.append(p2_action)

        # Store (state, action, reward, next_state) to Replay Buffer for Reinforcement Learning
        if len(p1_state_deque) == multi_step or done:
            n_reward = multi_step_reward(p1_reward_deque, gamma)
            n_state = p1_state_deque[0]
            n_action = p1_action_deque[0]
            p1_replay_buffer.push(n_state, n_action, n_reward, p1_next_state, np.float32(done))

            n_reward = multi_step_reward(p2_reward_deque, gamma)
            n_state = p2_state_deque[0]
            n_action = p2_action_deque[0]
            p2_replay_buffer.push(n_state, n_action, n_reward, p2_next_state, np.float32(done))
        
        # Store (state, action) to Reservoir Buffer for Supervised Learning
        if is_best_response:
            p1_reservoir_buffer.push(p1_state, p1_action)
            p2_reservoir_buffer.push(p2_state, p2_action)

        (p1_state, p2_state) = (p1_next_state, p2_next_state)

        # Logging
        p1_episode_reward += p1_reward
        p2_episode_reward += p2_reward
        tag_interval_length += 1


        # Episode done. Reset environment and clear logging records
        if done:
            env.reset()
            (p1_state, p2_state) = env.get_status()
            length_list.append(tag_interval_length)
            tag_interval_length = 0
            p1_reward_list.append(p1_episode_reward)
            p2_reward_list.append(p2_episode_reward)
            p1_episode_reward, p2_episode_reward, tag_interval_length = 0, 0, 0
            p1_state_deque.clear(), p2_state_deque.clear()
            p1_reward_deque.clear(), p2_reward_deque.clear()
            p1_action_deque.clear(), p2_action_deque.clear()

        if (len(p1_replay_buffer) > rl_start and
            len(p1_reservoir_buffer) > sl_start and
            frame_idx % train_freq == 0):

            # Update Best Response with Reinforcement Learning
            loss = compute_rl_loss(p1_current_model, p1_target_model, p1_replay_buffer, p1_rl_optimizer)
            p1_rl_loss_list.append(loss.item())

            loss = compute_rl_loss(p2_current_model, p2_target_model, p2_replay_buffer, p2_rl_optimizer)
            p2_rl_loss_list.append(loss.item())

            # Update Average Strategy with Supervised Learning
            loss = compute_sl_loss(p1_policy, p1_reservoir_buffer, p1_sl_optimizer)
            p1_sl_loss_list.append(loss.item())

            loss = compute_sl_loss(p2_policy, p2_reservoir_buffer, p2_sl_optimizer)
            p2_sl_loss_list.append(loss.item())

        

        if frame_idx % n_update_target == 0:
            update_target(p1_current_model, p1_target_model)
            update_target(p2_current_model, p2_target_model)

        if frame_idx % evaluation_interval == 0:
            p1_ratio.append(np.mean(pk_ai_sim(p1_policy)))
            p2_ratio.append(np.mean(pk_ai_sim(p2_policy)))
        # Logging and Saving models
        # if frame_idx % evaluation_interval == 0:
        #     p1_reward_list.clear(), p2_reward_list.clear(), length_list.clear()
        #     p1_rl_loss_list.clear(), p2_rl_loss_list.clear()
        #     p1_sl_loss_list.clear(), p2_sl_loss_list.clear()

        
    return {
        'p1_model':[p1_current_model,p1_policy],
        'p2_model':[p2_current_model,p2_policy],
        'p1_loss':[p1_rl_loss_list,p1_sl_loss_list],
        'p2_loss':[p2_rl_loss_list,p2_sl_loss_list],
        'p1_reward':p1_reward_list,
        'p2_reward':p2_reward_list,
        'len':length_list,
        'ratio':[p1_ratio,p2_ratio]
    }


def compute_sl_loss(policy, reservoir_buffer, optimizer):
    state, action = reservoir_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    action = torch.LongTensor(action).to(device)

    probs = policy(state)
    probs_with_actions = probs.gather(1, action.unsqueeze(1))
    log_probs = probs_with_actions.log()

    loss = -1 * log_probs.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_rl_loss(current_model, target_model, replay_buffer, optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    weights = torch.ones(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)
    weights = torch.FloatTensor(weights).to(device)

    # Q-Learning with target network
    q_values = current_model(state)
    target_next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.max(1)[0]
    expected_q_value = reward + (gamma ** multi_step) * next_q_value * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret