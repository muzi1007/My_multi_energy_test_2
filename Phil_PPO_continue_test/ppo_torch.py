import numpy as np
import torch as T
from buffer import PPOMemory
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(input_dims, alpha, n_actions)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

        self.reparam_noise = 1e-6
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mean, std = self.actor(state)
        action_dist = T.distributions.Normal(mean, std)
        actions = action_dist.sample()
        action = (T.tanh(actions)).to(self.device)
        value = self.critic(state)
        log_probs = action_dist.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(0, keepdim=True)
        log_probs = log_probs.cpu().detach().numpy()[0]
        value = T.squeeze(value).item()
        action = action.cpu().detach().numpy()[0]
        action = action * 2

        return action, log_probs, value


    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                actions = actions/2
                mean, std = self.actor(states)
                action_dist = T.distributions.Normal(mean, std)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)
                #action = (T.tanh(actions)).to(self.device)
                action = actions.to(self.device)
                new_probs = action_dist.log_prob(actions)
                new_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
                #prob_ratio = new_probs.exp() / old_probs.exp()
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


