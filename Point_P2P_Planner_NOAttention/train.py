# from PlanNet import *
# from Environment import *
# from DDPG import *

# import torch
# import torch.nn as nn
# import numpy as np

# if __name__ == "__main__":
#     env = Environment(
#         Polygon(Point2D(0,0), Point2D(0, 10), Point2D(10, 10) ,Point2D(10, 0)),
#         [
#             # Polygon(Point2D(1,1), Point2D(1, 2), Point2D(2, 2) ,Point2D(2, 1))
#         ],
#         None,
#         [0.5, 0.5, 0.2],
#         Point3D(0.5, 0.5, 0),
#         Point3D(9.5, 9.5, 0)
#     )
#     static_feature = []
#     for f in env.scene.get_features():
#         for v in f:
#             static_feature.append(float(v))
#     for f in env.goal_feature:
#         static_feature.append(float(f))
#     static_feature_torch = torch.tensor(static_feature, dtype=torch.float32)
    
#     state = env.reset()
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     action_size = 2
#     action_range = [0.5, 0.5]
#     buffer_size = int(1e5)
#     batch_size = 32
#     gamma = 0.9
#     tau = 0.1
#     lr_actor = 1e-6
#     lr_critic = 1e-6
#     update_every = 20
#     critic_local = Critic(input_dim=5, output_dim=1, feature_dim=128, num_layers=3, other_features=static_feature_torch.to(device))
#     critic_target = Critic(input_dim=5, output_dim=1, feature_dim=128, num_layers=3, other_features=static_feature_torch.to(device))
#     actor_local = Actor(input_dim=3, output_dim=action_size, feature_dim=128, num_layers=3, other_features=static_feature_torch.to(device))
#     actor_target = Actor(input_dim=3, output_dim=action_size, feature_dim=128, num_layers=3, other_features=static_feature_torch.to(device))
#     agent = DDPG(
#         critic_local=critic_local,
#         critic_target=critic_target,
#         actor_local=actor_local,
#         actor_target=actor_target,
#         buffer_size=buffer_size,
#         batch_size=batch_size,
#         gamma=gamma,
#         tau=tau,
#         lr_actor=lr_actor,
#         lr_critic=lr_critic,
#         action_size=action_size,
#         action_range=action_range,
#         env=env,
#         update_every=update_every,
#         device=device
#     )
#     while agent.episode < 20000:
#         agent.t_step = 0
#         env.reset()
#         agent.state = torch.from_numpy(np.array(env.robot.get_state_features())).float().to(device)
#         agent.reward_sum = 0
#         agent.done = False
#         agent.learn()
        
#         if agent.episode % 100 == 0:        
#             # test
#             plt.figure()
#             ax = plt.gca()
#             env.reset()
#             nmodel = copy.deepcopy(agent.actor_local).to(device)
#             nmodel.eval()
#             while True:
#                 state = torch.from_numpy(np.array(env.robot.get_state_features())).float().to(device)
#                 state.unsqueeze_(0)
#                 with torch.no_grad():
#                     action = nmodel(state).cpu().squeeze_(0).data.numpy()
#                 print("Action:", action)
#                 _, _, done, _ = env.step(action)
#                 env.plot(ax)
#                 plt.pause(0.1)
#                 if done:
#                     plt.close('all')
#                     break
#     agent.actor_local = agent.actor_local.to('cpu')
#     torch.save(agent.actor_local.state_dict(), "actor_local.pth")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from Environment import *
from PlanNet import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

gamma = 0.9
update_iteration = 200
batch_size = 64
tau = 0.05

max_episode=100000
exploration_noise = 0.1

class Replay_buffer:
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X))
            y.append(np.array(Y))
            u.append(np.array(U))
            r.append(np.array(R))
            d.append(np.array(D))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class DDPG(object):
    def __init__(self, state_dim, action_dim, other_features=None, max_action=1):
        self.actor = Actor(state_dim, action_dim, other_features = other_features).to(device)
        self.actor_target = Actor(state_dim, action_dim, other_features = other_features).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim + action_dim, 1, other_features=other_features).to(device)
        self.critic_target = Critic(state_dim + action_dim, 1, other_features=other_features).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze_(0).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

if __name__ == "__main__":
    env = Environment(
        Polygon(Point2D(0,0), Point2D(0, 10), Point2D(10, 10) ,Point2D(10, 0)),
        [
            # Polygon(Point2D(1,1), Point2D(1, 2), Point2D(2, 2) ,Point2D(2, 1))
        ],
        None,
        [0.5, 0.5, 0.2],
        Point3D(0.5, 0.5, 0),
        Point3D(9.5, 9.5, 0)
    )
    static_feature = []
    for f in env.scene.get_features():
        for v in f:
            static_feature.append(float(v))
    for f in env.goal_feature:
        static_feature.append(float(f))
    static_feature_torch = torch.tensor(static_feature, dtype=torch.float32)
    
    agent = DDPG(3, 2, other_features=static_feature_torch.to(device))
    ep_r = 0
    total_step = 0
    for i in range(max_episode):
        total_reward = 0
        step =0
        env.reset()
        env.step([0,0])
        state = np.array(env.robot.get_state_features())
        for t in count():
            action = agent.select_action(state)
            action = (action + np.random.normal(0, exploration_noise, size=2))

            next_state, reward, done, info = env.step(action) 
            agent.replay_buffer.push((state, next_state, action, reward, np.array(float(done))))

            state = next_state
            
            step += 1
            total_reward += reward
            
            if done:
                break
        total_step += step
        print("Total T:{} Episode: \t{} Total Reward: \t{:0.3f}".format(total_step, i, total_reward))
        agent.update()