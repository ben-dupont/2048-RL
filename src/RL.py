import os
import numpy as np
import pandas as pd
import random
import yaml
import pickle
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

        # Define policy and target networks
        self.policy_network = network(self.env.observation_space).to(device)
        self.target_network = network(self.env.observation_space).to(device)

        # Initialize policy and target networks with trunctated normal distribution
        self.policy_network.apply(network.truncated_normal_init)
        self.target_network.apply(network.truncated_normal_init)

    def load_network(self, network_dict_path):
        """
        Load the policy and target networks' state dictionaries from a specified file.

        Args:
            network_dict_path (str): The file path to the saved state dictionary.

        Returns:
            None
        """
        self.policy_network.load_state_dict(torch.load(network_dict_path))
        self.target_network.load_state_dict(torch.load(network_dict_path))

    def eval(self):
        """
        Sets the policy network to evaluation mode, which affects
        certain layers like dropout and batch normalization that behave differently
        during training and evaluation.
        """
        self.policy_network.eval()
    
    def train(self):
        """
        Sets the policy network to training mode, which affects
        certain layers like dropout and batch normalization that behave differently
        during training and evaluation.
        """
        self.policy_network.train()

    def select_greedy_action(self, state, device="mps", verbose=False):
        """
        Predict the best action for a given state using the greedy policy network.
        Args:
            state (np.array): The current state of the environment.
            device (str, optional): The device to run the computations on. Default is "mps".
            verbose (bool, optional): If True, additional information will be printed. Default is False.
        Returns:
            tuple: A tuple containing:
                - best_action (int): The index of the best action to take.
                - evaluate[best_action] (float): The value of the best action.
        """
        # Initialize afterstates and rewards
        afterstates = []
        rewards = []

        # Get allowed actions of this state
        isAllowed = self.env.unwrapped.allowed_actions()
        
        # Evaluate all possible afterstates and rewards
        for action in range(self.env.action_space.n):
            afterstate, reward = self.env.unwrapped.afterstate(state, action)
            afterstates.append(afterstate)
            if isAllowed[action]:
                rewards.append(reward)
            else:
                rewards.append(-np.Inf) # Assign negative infinity to forbidden actions
        
        # Evaluate values of all afterstates
        afterstates_tensor = torch.tensor(np.array(afterstates)).to(device)
        afterstate_values = self.policy_network.forward(afterstates_tensor).squeeze(1).detach().cpu().numpy()
        evaluate = np.array(rewards) + afterstate_values
        
        # Get the best action
        best_action = np.argmax(evaluate)
        
        return best_action, evaluate[best_action]

    def learn(self, total_timesteps, learning_rate=0.0005, gamma=0.99, buffer_size=50000, exploration_initial_eps=1.0, 
        exploration_final_eps=0.02, exploration_fraction=0.1, train_freq=1, batch_size=32, target_network_update_freq=500, 
        prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_increment=1e-4, 
        prioritized_replay_eps=1e-6, prioritized_replay_update_freq=500, checkpoint_path='./save_dir', checkpoint_freq=500, 
        check_gradient_freq=500, restart_jump=False, use_symmetry=False):
        """
        Train the reinforcement learning agent.
        Parameters:
        - total_timesteps (int): Total number of timesteps to train the agent.
        - learning_rate (float): Learning rate for the optimizer.
        - gamma (float): Discount factor for future rewards.
        - buffer_size (int): Size of the replay buffer.
        - exploration_initial_eps (float): Initial value of epsilon for epsilon-greedy action selection.
        - exploration_final_eps (float): Final value of epsilon for epsilon-greedy action selection.
        - exploration_fraction (float): Fraction of total timesteps over which epsilon is annealed.
        - train_freq (int): Frequency of training steps.
        - batch_size (int): Batch size for training.
        - target_network_update_freq (int): Frequency of updates to the target network.
        - prioritized_replay (bool): Whether to use prioritized experience replay.
        - prioritized_replay_alpha (float): Alpha parameter for prioritized replay.
        - prioritized_replay_beta0 (float): Initial value of beta for prioritized replay.
        - prioritized_replay_beta_increment (float): Increment value for beta in prioritized replay.
        - prioritized_replay_eps (float): Epsilon value for prioritized replay.
        - prioritized_replay_update_freq (int): Frequency of updates to the priorities in the replay buffer.
        - checkpoint_path (str): Path to save checkpoints.
        - checkpoint_freq (int): Frequency of saving checkpoints.
        - check_gradient_freq (int): Frequency of checking gradients.
        - restart_jump (bool): Whether to restart the environment with a jump.
        - use_symmetry (bool): Whether to use symmetry in the environment.
        Returns: 
        None
        """
        
        # Create folder for checkpoints
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save training parameters to folder
        training_parameters = {
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "buffer_size": buffer_size,
            "exploration": {
                "initial_eps": exploration_initial_eps,
                "final_eps": exploration_final_eps,
                "fraction": exploration_fraction
            },
            "train_freq": train_freq,
            "batch_size": batch_size,
            "target_network_update_freq": target_network_update_freq,
            "prioritized_replay": {
                "enabled": prioritized_replay,
                "prioritized_replay_alpha": prioritized_replay_alpha,
                "prioritized_replay_beta": prioritized_replay_beta0,
                "prioritized_replay_beta_increment": prioritized_replay_beta_increment,
                "prioritized_replay_eps": prioritized_replay_eps,
                "reprioritize_freq": prioritized_replay_update_freq,
            },
            "checkpoint": {
                "path": checkpoint_path,
                "freq": checkpoint_freq
            },
            "strategy":{
                "restart_jump": restart_jump,
                "use_symmetry": use_symmetry
            }
        }

        with open(checkpoint_path + "training_parameters.yaml", "w") as file:
            yaml.dump(training_parameters, file, default_flow_style=False)

        # Initialization
        optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        i_timestep = 0 
        i_episode = 0
        loss = np.array([]) # Array to store loss at each training step
        episode_rewards = np.array([]) # Array to store episode rewards
        episode_initial_estimated_value = np.array([]) # Array to store model estimation of episode value from first state
        max_tiles = np.array([]) # Array to store max tiles reached in each episode
        epsilons = np.array([])  # Array to store epsilon values at each timestep

        self.train()

        # Initialize replay buffer
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, self.env.observation_space, self.env.action_space, alpha=prioritized_replay_alpha, beta=prioritized_replay_beta0, beta_increment=prioritized_replay_beta_increment, device=self.device)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, self.env.observation_space, self.env.action_space, device=self.device)

        print(f"Replay buffer memory size: {asizeof.asizeof(self.replay_buffer) / 1024 ** 2:.2f} MB")

        # Training loop
        while i_timestep < total_timesteps:
            i_episode += 1
            firstMove = True
            
            state, _ = self.env.reset(jump=restart_jump)

            terminated = False
            episode_reward = 0
            
            while not terminated:
                i_timestep += 1

                # Epsilon-greedy action selection
                if i_timestep > exploration_fraction * total_timesteps:
                    eps_threshold = exploration_final_eps
                else:
                    eps_threshold = exploration_initial_eps - i_timestep / total_timesteps / exploration_fraction * (exploration_initial_eps - exploration_final_eps)
                epsilons = np.append(epsilons, eps_threshold)

                if random.random() > eps_threshold:
                    action, estimate_value = self.select_greedy_action(state)
                else:
                    allowed_actions = self.env.unwrapped.allowed_actions()
                    isAllowed = False
                    while not(isAllowed):
                        action = self.env.action_space.sample()
                        isAllowed = allowed_actions[action]

                # Take action and observe next state and reward
                afterstate, reward = self.env.unwrapped.afterstate(state, action)
                state, reward, terminated, _, _ = self.env.unwrapped.step(action)
                episode_reward += reward

                # Store state, after state and reward into replay buffer
                if not(firstMove): # previous_afterstate not defined for first move
                    self.replay_buffer.add(previous_afterstate, afterstate, np.array([action]), reward, False, np.array([{}]))
                    if terminated: # store final transition with 0 as reward, very important for TD(0) to work
                        self.replay_buffer.add(afterstate, afterstate, np.array([action]), 0, True, np.array([{}]))
                else: 
                    episode_initial_estimated_value = np.append(episode_initial_estimated_value, estimate_value)

                firstMove = False
                previous_afterstate = afterstate # store afterstate for next transition

                # Run optimization step
                check_gradient = i_timestep % check_gradient_freq == 0 and i_timestep > 1
                use_target_network = target_network_update_freq > 0
                if i_timestep % train_freq == 0:
                    i_loss = self._optimize_model(optimizer, batch_size=batch_size, gamma=gamma, use_target_network=use_target_network, device=self.device, check_gradient=check_gradient, timestep=i_timestep, symmetry=use_symmetry, save_path=checkpoint_path)
                    loss = np.append(loss, i_loss)

                # Update target network
                if i_timestep % target_network_update_freq == 0 and target_network_update_freq > 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
            
            # Store episode rewards and max tiles
            episode_rewards = np.append(episode_rewards, episode_reward)
            max_tiles = np.append(max_tiles, np.max(self.env.unwrapped.decode(state)))
            # Plot training progress
            plot_save = i_episode % checkpoint_freq == 0 and i_episode > 1
            self._plot_callback(loss, episode_rewards, episode_initial_estimated_value, max_tiles, epsilons, plot_save, checkpoint_path)
            # Save model checkpoint and logs
            if i_episode % checkpoint_freq == 0 and i_episode > 1:
                PATH = checkpoint_path + 'rl_model_%i_epsiodes.zip' % i_episode
                torch.save(self.policy_network.state_dict(), PATH)
                data = {
                    'loss': loss,
                    'episode_rewards': episode_rewards,
                    'episode_initial_estimated_value': episode_initial_estimated_value,
                    'max_tiles': max_tiles,
                    'epsilons': epsilons
                }
                with open(checkpoint_path + 'logs.pkl' % i_episode, 'w') as file:
                    pickle.dump(data, file)

    def _optimize_model(self, optimizer, batch_size=32, gamma=0.99, use_target_network=True, device="mps", check_gradient=False, timestep=0, symmetry=False, save_path='./'):
        """
        Optimize the model by performing a single step of gradient descent on the policy network.
        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use for updating the network parameters.
            batch_size (int, optional): The number of samples to use for each optimization step. Default is 32.
            gamma (float, optional): The discount factor for future rewards. Default is 0.99.
            use_target_network (bool, optional): Whether to use the target network for computing next state values. Default is True.
            device (str, optional): The device to use for computations ('cpu', 'cuda', 'mps'). Default is 'mps'.
            check_gradient (bool, optional): Whether to check the gradients during optimization. Default is False.
            timestep (int, optional): The current timestep, used for logging gradient checks. Default is 0.
            symmetry (bool, optional): Whether to use data augmentation by symmetry. Default is False.
            save_path (str, optional): The path to save gradient check logs. Default is './'.
        Returns:
            float: The loss value after the optimization step.
        """
        if self.replay_buffer.size() < batch_size:
            return

        # Sample from replay buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            replay_data, indices, weights = self.replay_buffer.sample(batch_size)
            weights = weights.to(device=device, dtype=torch.float32)
        else:
            replay_data = self.replay_buffer.sample(batch_size)
            weights = 1

        afterstates = replay_data.observations.to(device)
        next_afterstates = replay_data.next_observations.to(device)
        rewards = replay_data.rewards.to(device)
        dones = replay_data.dones.to(device)

        # Compute estimated and expected afterstates values
        with torch.no_grad():
            if use_target_network:
                next_afterstates_values = self.target_network.forward(next_afterstates)
            else:
                next_afterstates_values = self.policy_network.forward(next_afterstates)
        afterstates_target_values = (next_afterstates_values*gamma + rewards) * (dones == False)
        afterstates_current_values = self.policy_network.forward(afterstates)

        # Update priorities in prioritized replay buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            new_priorities = torch.abs(afterstates_target_values - afterstates_current_values).cpu().detach().numpy()
            self.replay_buffer.update_priorities(indices, new_priorities)

        # Data augmentation: 8 symmetric positions all have same expected values
        if symmetry:
            afterstates_current_values, afterstates_target_values, weights = self._symmetryAugmentation(afterstates, afterstates_current_values, afterstates_target_values, weights)

        # Computed weighted values (for prioritized experience replay)
        weighted_afterstates_current_values = weights * afterstates_current_values
        weighted_afterstates_target_values = weights * afterstates_target_values

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(weighted_afterstates_current_values, weighted_afterstates_target_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        if check_gradient:
            self._check_gradients(timestep, save_path=save_path, threshold_min=1e-6, threshold_max=1e2, verbose=False)
        
        optimizer.step()

        return loss.cpu().detach().numpy()

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

    def _symmetryAugmentation(self, states, states_current_values, states_target_values, weights):
        """
        Perform symmetry augmentation on the given states and their corresponding values and weights.

        Args:
            states (torch.Tensor): The original states tensor.
            states_current_values (torch.Tensor): The current values of the original states.
            states_target_values (torch.Tensor): The target values of the original states.
            weights (torch.Tensor or None): The weights for prioritized experience replay.

        Returns:
            tuple: A tuple containing:
                - states_current_values (torch.Tensor): The current values of the original and symmetric states.
                - states_target_values (torch.Tensor): The target values repeated for all symmetric states.
                - weights (torch.Tensor or None): The weights repeated and normalized for all symmetric states.
        """
        # Generate symmetric states
        states_rot90 = torch.rot90(states, k=1, dims=(-2, -1))
        states_rot180 = torch.rot90(states, k=2, dims=(-2, -1))
        states_rot270 = torch.rot90(states, k=3, dims=(-2, -1))
        states_flip = torch.flip(states, dims=[-2])
        states_rot90_flip = torch.flip(states_rot90, dims=[-2])
        states_rot180_flip = torch.flip(states_rot180, dims=[-2])
        states_rot270_flip = torch.flip(states_rot270, dims=[-2])

        # Evaluate values of symmetric states and merge with original states values
        rotated_states = torch.cat((states_rot90, states_rot180, states_rot270, states_flip, states_rot90_flip, states_rot180_flip, states_rot270_flip), dim=0)
        rotated_states_current_values = self.policy_network.forward(rotated_states)
        states_current_values = torch.cat((states_current_values, rotated_states_current_values))

        # Assign same target values and weights (for prioritized experience replay) for all symmetric states
        states_target_values = states_target_values.repeat(8, 1)
        if type(weights) == torch.Tensor:
            weights = weights.repeat(8)
            weights = weights / 8

        return states_current_values, states_target_values, weights

    def _plot_callback(self, loss, episode_rewards, episode_initial_estimated_value, max_tiles, epsilons, plot_save, save_path):
        """
        Plots various metrics related to the training of a reinforcement learning model.

        Parameters:
        - loss (list): A list of loss values recorded during training.
        - episode_rewards (list): A list of rewards obtained per episode.
        - episode_initial_estimated_value (list): A list of initial estimated values per episode.
        - max_tiles (list): A list of maximum tiles reached per episode.
        - epsilons (list): A list of epsilon values over timesteps.
        - plot_save (bool): A flag indicating whether to save the plot.
        - save_path (str): The path where the plot should be saved if plot_save is True.

        Returns:
        None
        """
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