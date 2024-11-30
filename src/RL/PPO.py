import os
import numpy as np
import pandas as pd
import random
import yaml
import pickle
import logging
from pympler import asizeof
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from src.buffers import PrioritizedReplayBuffer

class PPO():
    def __init__(self, model, env, shared_features_extractor=False, device="mps"):
        self.env = env
        self.device = device

        # Define policy and target networks
        self.model = model(self.env.observation_space, action_dim=self.env.action_space.n, shared_features_extractor=shared_features_extractor).to(device)

    def load_network(self, model_dict_path):
        """
        Load the policy and target networks' state dictionaries from a specified file.

        Args:
            actor_model_dict_path (str): The file path to the saved actor model state dictionary.
            critic_model_dict_path (str): The file path to the saved critic model state dictionary.

        Returns:
            None
        """
        self.model.load_state_dict(torch.load(model_dict_path))

    def eval(self):
        """
        Sets the policy network to evaluation mode, which affects
        certain layers like dropout and batch normalization that behave differently
        during training and evaluation.
        """
        self.model.eval()
    
    def train(self):
        """
        Sets the policy network to training mode, which affects
        certain layers like dropout and batch normalization that behave differently
        during training and evaluation.
        """
        self.model.train()

    def predict(self, state, device="mps"):
        state = torch.tensor(np.array([state])).to(device)
        action_probs, value = self.model.forward(state)
        action_probs = action_probs + 1e-8
        
        # Create a mask for allowed actions
        is_allowed = self.env.unwrapped.allowed_actions()
        mask = torch.tensor(is_allowed, dtype=torch.float32).to(device)
        
        # Apply mask using multiplication (non-in-place operation)
        masked_probs = action_probs * mask
        
        # Renormalize probabilities
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
        
        dist = torch.distributions.Categorical(masked_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, value, log_prob

    def learn(self, total_episodes, learning_rate=0.0005, gamma=0.99, gae_lambda=0.95, epsilon=0.2, vf_coef=0.5, n_epochs=10, batch_size=64, checkpoint_path='./save_dir', checkpoint_freq=500):
        """
        Train the agent using PPO algorithm.
        Parameters:
        - total_episodes (int): Total number of episodes to train
        - learning_rate (float): Learning rate for the optimizer
        - gamma (float): Discount factor for future rewards
        - gae_lambda (float): Lambda parameter for GAE
        - epsilon (float): Clipping parameter for PPO
        - vf_coef (float): Coefficient for value function loss
        - n_epochs (int): Number of epochs to optimize on the same trajectory
        - batch_size (int): Number of episodes to sample from the buffer
        - checkpoint_path (str): Path to save checkpoints
        - checkpoint_freq (int): Frequency of saving checkpoints
        """
        
        # Create folder for checkpoints
        os.makedirs(checkpoint_path, exist_ok=True)

        # Initialize logging
        logging.basicConfig(
            filename=checkpoint_path+'training.log',         # Log file name
            level=logging.INFO,              # Logging level
            format='%(asctime)s - %(message)s',  # Format for log messages
            datefmt='%Y-%m-%d %H:%M:%S'      # Date format
        )
        logging.info("Training started.")

        # Save training parameters to folder
        training_parameters = {
            "total_episodes": total_episodes,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "checkpoint": {
                "path": checkpoint_path,
                "checkpoint_freq": checkpoint_freq
            }
        }

        with open(checkpoint_path + "training_parameters.yaml", "w") as file:
            yaml.dump(training_parameters, file, default_flow_style=False)

        # Initialization
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        loss = np.array([]) # Array to store loss at each training step
        episode_rewards = np.array([]) # Array to store episode rewards
        max_tiles = np.array([]) # Array to store max tiles reached in each episode
        first_action_values = np.array([]) # Array to store first action values in each episode
        last_action_values = np.array([]) # Array to store last action values in each episode

        self.train()

        # Training loop
        for i_episode in range(total_episodes):
            state, _ = self.env.reset(jump=False)
            terminated = False

            # Lists to store episode data
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            
            # Collect trajectory
            while not terminated:
                state_tensor = torch.tensor(np.array([state])).to(self.device)
                with torch.no_grad():
                    action_probs, value = self.model.forward(state_tensor)
                    
                    # Create a mask for allowed actions
                    is_allowed = self.env.unwrapped.allowed_actions()
                    mask = torch.tensor(is_allowed, dtype=torch.float32).to(self.device)
                    
                    # Apply mask and renormalize
                    masked_probs = action_probs * mask
                    masked_probs = masked_probs + 1e-8
                    masked_probs = masked_probs / masked_probs.sum()
                    
                    dist = torch.distributions.Categorical(masked_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                # Take action and observe next state and reward
                next_state, reward, terminated, _, _ = self.env.unwrapped.step(action.item())
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                
                state = next_state

            # Store first and last action values, episode rewards, and max tiles for plotting
            first_action_values = np.append(first_action_values, values[0].item())
            last_action_values = np.append(last_action_values, values[-1].item())
            episode_rewards = np.append(episode_rewards, np.sum(rewards))
            max_tiles = np.append(max_tiles, np.max(self.env.unwrapped.decode(state)))
            # Compute returns and advantages using GAE
            gae = 0
            advantages = []
            returns = []
            next_value = 0  # For terminal state
            
            for reward, value in zip(reversed(rewards), reversed(values)):
                delta = reward + gamma * next_value - value.item()
                gae = delta + gamma * gae_lambda * gae
                advantage = gae
                returns.insert(0, advantage + value.item())
                advantages.insert(0, advantage)
                next_value = value.item()
            
            # Convert to tensors
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            actions = torch.stack(actions)
            old_log_probs = torch.stack(log_probs).detach()
            values = torch.cat(values)

            # PPO update for n_epochs
            for _ in range(n_epochs):
                # Sample batch
                indices = np.arange(len(states))
                np.random.shuffle(indices)
                mini_batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

                for mini_batch in mini_batches:
                    # Get current policy and value predictions
                    action_probs, current_values = self.model.forward(states[mini_batch])
                    masked_probs = []

                    # Create masks for allowed actions
                    for k in range(len(states[mini_batch])):
                        is_allowed = self.env.unwrapped.allowed_actions(state=states[mini_batch][k].cpu().numpy())
                        mask = torch.tensor(is_allowed, dtype=torch.float32).to(self.device)
                        
                        # Apply mask and renormalize
                        masked_prob = action_probs[k] * mask
                        masked_prob = masked_prob + 1e-8
                        masked_prob = masked_prob / masked_prob.sum()
                        masked_probs.append(masked_prob)
                    masked_probs = torch.stack(masked_probs)

                    dist = torch.distributions.Categorical(masked_probs)
                    current_log_probs = dist.log_prob(actions[mini_batch].squeeze(1))

                    # Normalize advantages at mini-batch level
                    advantages[mini_batch] = (advantages[mini_batch] - advantages[mini_batch].mean()) / (advantages[mini_batch].std() + 1e-8)

                    # Calculate ratios and surrogate losses
                    ratios = torch.exp(current_log_probs - old_log_probs[mini_batch].squeeze(1))
                    surr1 = ratios * advantages[mini_batch]
                    surr2 = torch.clamp(ratios, 1-epsilon, 1+epsilon) * advantages[mini_batch]

                    # Calculate actor and critic losses
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = vf_coef * (returns[mini_batch] - current_values.squeeze(1)).pow(2).mean()
                    
                    # Combined loss
                    total_loss = actor_loss + critic_loss
                    
                    # Optimization step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    loss = np.append(loss, total_loss.item())
            
            # Plot training progress
            plot_save = i_episode % checkpoint_freq == 0 and i_episode > 1
            self._plot_callback(loss, episode_rewards, max_tiles, first_action_values, last_action_values, plot_save, checkpoint_path)
            # Save model checkpoint and logs
            if i_episode % checkpoint_freq == 0 and i_episode > 1:
                logging.info(f"Episode {i_episode} - Reward: {episode_rewards[-1]} - Max Tile: {max_tiles[-1]}")
                PATH = checkpoint_path + 'rl_model_%i_epsiodes.zip' % i_episode
                torch.save(self.model.state_dict(), PATH)
                data = {
                    'loss': loss,
                    'episode_rewards': episode_rewards,
                    'max_tiles': max_tiles,
                }
                with open(checkpoint_path + 'logs.pkl', 'wb') as file:
                    pickle.dump(data, file)

    def _check_gradients(self, timestep, save_path='./', threshold_min=1e-6, threshold_max=1e2, verbose=False):
        """
        Check the gradients of the policy network for vanishing or exploding values.
        Args:
            timestep (int): The current timestep in the training process.
            save_path (str, optional): The path where the gradient warnings will be saved. Defaults to './'.
            threshold_min (float, optional): The minimum threshold for gradient norms to detect vanishing gradients. Defaults to 1e-6.
            threshold_max (float, optional): The maximum threshold for gradient norms to detect exploding gradients. Defaults to 1e2.
            verbose (bool, optional): If True, prints the gradient norms for each layer. Defaults to False.
        Returns:
            None
        """
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

    def _plot_callback(self, loss, episode_rewards, max_tiles, first_action_values, last_action_values, plot_save, save_path):
        """
        Plots various metrics related to the training of a reinforcement learning model.

        Parameters:
        - loss (list): A list of loss values recorded during training.
        - episode_rewards (list): A list of rewards obtained per episode.
        - max_tiles (list): A list of maximum tiles reached per episode.
        - plot_save (bool): A flag indicating whether to save the plot.
        - save_path (str): The path where the plot should be saved if plot_save is True.

        Returns:
        None
        """
        clear_output(wait=True)  # Clear the previous plot

        # Create the subplot layout
        fig, axes = plt.subplots(4, 1, figsize=(8, 16))
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

        # First action value plot (third row, second column)
        axes[3].plot(first_action_values, color="blue", label="First Action Value")
        axes[3].plot(last_action_values, color="red", label="Last Action Value")
        axes[3].set_title("First and Last Action Value")
        axes[3].set_xlabel("Episode")
        axes[3].set_ylabel("Action Value")
        axes[3].legend()
        axes[3].grid()

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Save
        if plot_save:
            fig.savefig(save_path+'logs.png', dpi=300, bbox_inches='tight')