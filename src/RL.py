import os
import numpy as np
import pandas as pd
import random
import json
from pympler import asizeof
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from src.buffers import PrioritizedReplayBuffer

class TDL():
    def __init__(self, network, env, device="mps"):
        self.env = env
        self.device = device

        self.policy_network = network(self.env.observation_space).to(device)
        self.target_network = network(self.env.observation_space).to(device)

        self.policy_network.apply(network.truncated_normal_init)
        self.target_network.apply(network.truncated_normal_init)

    def load_network(self, network_dict_path):
        self.policy_network.load_state_dict(torch.load(network_dict_path))
        self.target_network.load_state_dict(torch.load(network_dict_path))

    def predict(self, observation, device="mps", verbose=False):
        # Generate all afterstates in a list
        afterstates = []
        rewards = []

        isAllowed = self.env.unwrapped.allowed_actions()
        
        for action in range(self.env.action_space.n):
            afterstate, reward = self.env.unwrapped.afterstate(observation, action)
            afterstates.append(afterstate)
            if isAllowed[action]:
                rewards.append(reward)
            else:
                rewards.append(-np.Inf)
        
        # Convert afterstates to a tensor and move to the device in one go
        afterstates_tensor = torch.tensor(np.array(afterstates)).to(device)
        
        # Forward pass for all afterstates at once
        afterstate_values = self.policy_network.forward(afterstates_tensor).squeeze(1).detach().cpu().numpy()
        
        # Combine rewards and afterstate values
        evaluate = np.array(rewards) + afterstate_values
        
        # Get the best action
        best_action = np.argmax(evaluate)

        ## DEBUG
        if verbose:
            print("State\n")
            print(self.env.decode(observation))

            for action in range(self.env.action_space.n):
                print("Action: "+str(action))
                print("Afterstate")
                print(self.env.decode(afterstates[action]))
                print("Reward: "+str(rewards[action]))
                print("Afterstate value: "+str(afterstate_values[action]))
            
            print("Best action: " + str(best_action))
            print("Evaluate: " + str(evaluate[best_action]))
        
        return best_action, evaluate[best_action]

    def eval(self):
        self.policy_network.eval()
        return
    
    def train(self):
        self.policy_network.train()
        return

    def check_gradients(self, timestep, save_path='./', threshold_min=1e-6, threshold_max=1e2, verbose=False):
        warnings = []  # Collect warnings to minimize print calls
        
        # Create a list of norms in one pass
        norms = [(name, param.grad.data.norm(2).item()) 
                for name, param in self.policy_network.named_parameters() 
                if param.grad is not None]
        
        for name, grad_norm in norms:
            if grad_norm < threshold_min:
                warnings.append(f"Warning: Vanishing gradient in layer {name}, norm: {grad_norm}")
            elif grad_norm > threshold_max:
                warnings.append(f"Warning: Exploding gradient in layer {name}, norm: {grad_norm}")
            
            if verbose:
                print(f"Gradient in layer {name}, norm: {grad_norm}")

        # Print all warnings at once
        if warnings:
            with open(save_path+"gradients_log.txt", "a") as file:
                file.writelines(("\n Step " + str(timestep) + "\n"))
                file.writelines(("\n").join(warnings))

    def plot_callback(self, loss, episode_rewards, episode_initial_estimated_value, max_tiles, epsilons, plot_save, save_path):
        clear_output(wait=True)  # Clear the previous plot

        # Create the subplot layout
        fig, axes = plt.subplots(5, 1, figsize=(8, 16))
        fig.subplots_adjust(hspace=0.4)  # Adjust space between plots

        # Reward per episode plot (first row, spans both columns)
        axes[0].semilogy(loss, label="Loss", color="black")
        axes[0].set_title("Loss per Training Step")
        axes[0].set_xlabel("Training step")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid()

        # Reward per episode plot (first row, spans both columns)
        axes[1].plot(episode_rewards, label="Episode Reward")
        if len(episode_rewards) >= 100:
            df1 = pd.DataFrame({'episode_rewards':episode_rewards})
            df1['rolling_avg'] = df1['episode_rewards'].rolling(window=100).mean()
            axes[1].plot(df1['rolling_avg'], color="orange")
        axes[1].set_title("Reward per Episode")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Reward")
        axes[1].legend()
        axes[1].grid()

        # Max score reached plot (second row, second column)
        axes[2].plot(max_tiles, 'o', color="red", label="Max Score")
        if len(episode_rewards) >= 100:
            df2 = pd.DataFrame({'max_tiles':max_tiles})
            df2['rolling_avg'] = df2['max_tiles'].rolling(window=100).mean()
            axes[2].plot(df2['rolling_avg'], color="orange")
        axes[2].set_title("Max Tile Reached")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Max Tile")
        axes[2].legend()
        axes[2].grid()

        # Estimated value per episode plot
        axes[3].plot(episode_initial_estimated_value, label="Episode Estimated Value")
        if len(episode_rewards) >= 100:
            df3 = pd.DataFrame({'episode_estimated_value':episode_initial_estimated_value})
            df3['rolling_avg'] = df3['episode_estimated_value'].rolling(window=100).mean()
            axes[3].plot(df3['rolling_avg'], color="orange")
        axes[3].set_title("Estimated value per Episode")
        axes[3].set_xlabel("Episode")
        axes[3].set_ylabel("Estimated Value")
        axes[3].legend()
        axes[3].grid()

        # Epsilon threshold plot (second row, first column)
        axes[4].plot(epsilons, color="orange", label="Epsilon")
        axes[4].set_title("Epsilon Threshold Over Timesteps")
        axes[4].set_xlabel("Timestep")
        axes[4].set_ylabel("Epsilon")
        axes[4].set_ylim(0, 1.0)
        axes[4].legend()
        axes[4].grid()

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Save
        if plot_save:
            fig.savefig(save_path+'logs.png', dpi=300, bbox_inches='tight')

    def symmetries(self, states, expected_states_values, weights):
        states_rot90 = torch.rot90(states, k=1, dims=(-2, -1))
        states_rot180 = torch.rot90(states, k=2, dims=(-2, -1))
        states_rot270 = torch.rot90(states, k=3, dims=(-2, -1))
        states_flip = torch.flip(states, dims=[-2])
        states_rot90_flip = torch.flip(states_rot90, dims=[-2])
        states_rot180_flip = torch.flip(states_rot180, dims=[-2])
        states_rot270_flip = torch.flip(states_rot270, dims=[-2])

        states = torch.cat((states, states_rot90, states_rot180, states_rot270, states_flip, states_rot90_flip, states_rot180_flip, states_rot270_flip), dim=0)
        expected_states_values = torch.cat((expected_states_values, expected_states_values, expected_states_values, expected_states_values, expected_states_values, expected_states_values, expected_states_values, expected_states_values), dim=0)
        if type(weights) == torch.Tensor:
            weights = torch.cat((weights, weights, weights, weights, weights, weights, weights, weights))
            weights = weights / 8
        return states, expected_states_values, weights

    def optimize_model(self, optimizer, lr=0.0001, gamma=0.99, use_target_network=True, device="mps", check_gradient=False, timestep=0, symmetry=False, save_path='./'):
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample from replay buffer
        if self.prioritized_replay:
            replay_data, weights = self.replay_buffer.sample(self.batch_size)
            weights = weights.to(device)
        else:
            replay_data = self.replay_buffer.sample(self.batch_size)
            weights = 1

        afterstates = replay_data.observations.to(device)
        next_afterstates = replay_data.next_observations.to(device)
        rewards = replay_data.rewards.to(device)
        dones = replay_data.dones.to(device)

        # Compute loss
        with torch.no_grad():
            if use_target_network:
                next_afterstates_values = self.target_network.forward(next_afterstates)
            else:
                next_afterstates_values = self.policy_network.forward(next_afterstates)
        afterstates_target_values = (next_afterstates_values*gamma + rewards) * (dones == False)

        # Data augmentation: 8 symmetric positions all have same expected values
        if symmetry:
            afterstates, afterstates_target_values, weights = self.symmetries(afterstates, afterstates_target_values, weights)

        afterstates_current_values = self.policy_network.forward(afterstates)

        weighted_afterstates_target_values = weights * afterstates_target_values
        weighted_afterstates_current_values = weights * afterstates_current_values

        criterion = nn.MSELoss()
        loss = criterion(weighted_afterstates_current_values, weighted_afterstates_target_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # Check gradients
        if check_gradient:
            self.check_gradients(timestep, save_path=save_path, threshold_min=1e-6, threshold_max=1e2, verbose=False)
        # In-place gradient clipping
        #torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.cpu().detach().numpy()

    def learn(self, total_timesteps, lr=0.0001, batch_size=512, train_freq=1, buffer_size=50000, prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta=0.4, prioritized_replay_beta_increment=1e-4, reprioritize_freq=500, eps_start=0.9, eps_end=0.02, exploration_fraction=0.1, gamma=0.99, target_network_update_freq=500, save_path='./save_dir', save_freq=500, check_gradient_freq=500, jump=False, symmetry=False):
        # Create folder for training
        os.makedirs(save_path, exist_ok=True)

        # Save training parameters to folder
        data = {
            "total_timesteps": total_timesteps,
            "lr": lr,
            "batch_size": batch_size,
            "train_freq": train_freq,
            "buffer_size": buffer_size,
            "prioritized_replay": prioritized_replay,
            "prioritized_replay_alpha": prioritized_replay_alpha,
            "prioritized_replay_beta": prioritized_replay_beta,
            "prioritized_replay_beta_increment": prioritized_replay_beta_increment,
            "reprioritize_freq": reprioritize_freq,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "exploration_fraction": exploration_fraction,
            "gamma": gamma,
            "target_network_update_freq": target_network_update_freq,
            "symmetry": symmetry,
            "jump": jump
        }

        with open(save_path + "training_parameters.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

        optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        i_timestep = 0 
        i_episode = 0
        loss = np.array([])
        episode_rewards = np.array([])
        episode_initial_estimated_value = np.array([])
        max_tiles = np.array([])
        epsilons = np.array([]) 

        self.train()

        self.batch_size = batch_size
        self.prioritized_replay = prioritized_replay
        self.reprioritize_freq = reprioritize_freq
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, self.env.observation_space, self.env.action_space, alpha=prioritized_replay_alpha, beta=prioritized_replay_beta, beta_increment=prioritized_replay_beta_increment, device=self.device)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, self.env.observation_space, self.env.action_space, device=self.device)

        replay_buffer_size = asizeof.asizeof(self.replay_buffer)
        print(f"Accurate replay buffer memory size: {replay_buffer_size / 1024 ** 2:.2f} MB")

        while i_timestep < total_timesteps:
            i_episode += 1
            firstMove = True
            
            state, _ = self.env.reset(jump=jump)

            #max_tiles = np.append(max_tiles, - np.max(self.env.unwrapped.decode(state)))
            terminated = False
            episode_reward = 0
            
            while not terminated:
                i_timestep += 1

                if i_timestep > exploration_fraction * total_timesteps:
                    eps_threshold = eps_end
                else:
                    eps_threshold = eps_start - i_timestep / total_timesteps / exploration_fraction * (eps_start - eps_end)
                epsilons = np.append(epsilons, eps_threshold)

                if random.random() > eps_threshold:
                    action, estimate_value = self.predict(state)
                else:
                    allowed_actions = self.env.unwrapped.allowed_actions()
                    isAllowed = False
                    while not(isAllowed):
                        action = self.env.action_space.sample()
                        isAllowed = allowed_actions[action]

                next_afterstate, reward = self.env.unwrapped.afterstate(state, action)
                state, reward, terminated, _, _ = self.env.unwrapped.step(action)
                episode_reward += reward

                # Store state, after state and reward into replay buffer
                if not(firstMove):
                    self.replay_buffer.add(afterstate, next_afterstate, np.array([action]), reward, False, np.array([{}]))
                    if terminated:
                        self.replay_buffer.add(next_afterstate, next_afterstate, np.array([action]), 0, True, np.array([{}]))
                else: 
                    episode_initial_estimated_value = np.append(episode_initial_estimated_value, estimate_value)

                firstMove = False
                afterstate = next_afterstate

                # Optimize model
                check_gradient = i_timestep % check_gradient_freq == 0 and i_timestep > 1
                use_target_network = target_network_update_freq > 0
                if i_timestep % train_freq == 0:
                    i_loss = self.optimize_model(optimizer, lr=lr, gamma=gamma, use_target_network=use_target_network, check_gradient=check_gradient, timestep=i_timestep, save_path=save_path)
                    loss = np.append(loss, i_loss)

                # Update target network
                if i_timestep % target_network_update_freq == 0 and target_network_update_freq > 0:
                    target_net_state = ep_dict = self.target_network.state_dict()
                    policy_net_state_dict = self.policy_network.state_dict()
                    for key in policy_net_state_dict:
                        self.target_net_state_dict[key] = policy_net_state_dict[key]
                    self.target_network.load_state_dict(self.target_net_state_dict)

                # Update priorities of prioritized replay buffer
                if self.prioritized_replay:
                    if i_timestep % self.reprioritize_freq == 0:
                        self.replay_buffer.update_priorities(self.policy_network, self.target_network)
            
            episode_rewards = np.append(episode_rewards, episode_reward)
            #max_tiles[i_episode-1] += np.max(self.env.unwrapped.decode(state))
            max_tiles = np.append(max_tiles, np.max(self.env.unwrapped.decode(state)))
            # Plot episode reward
            plot_save = i_episode % save_freq == 0 and i_episode > 1
            self.plot_callback(loss, episode_rewards, episode_initial_estimated_value, max_tiles, epsilons, plot_save, save_path)
            # Save model checkpoint
            if i_episode % save_freq == 0 and i_episode > 1:
                PATH = save_path + 'rl_model_%i_epsiodes.zip' % i_episode
                torch.save(self.policy_network.state_dict(), PATH)