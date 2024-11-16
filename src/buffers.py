import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    A replay buffer that prioritizes certain experiences over others, based on their TD-error.
    Attributes:
        alpha (float): Controls how much prioritization is used (0 is uniform, 1 is full prioritization).
        beta (float): Controls how much importance-sampling correction is used.
        beta_increment (float): Rate at which beta increases to 1.
        eps (float): Small value to ensure non-zero priorities.
        rank_based (bool): If True, use rank-based prioritization; otherwise, use proportional prioritization.
        priorities (np.ndarray): Array to store the priorities of the experiences.
    Methods:
        add(*args, **kwargs):
            Add a transition with the maximum priority.
        sample(batch_size):
            Sample a batch of experiences based on their priorities.
        update_priorities(policy_network, target_network):
            Update the priorities of the experiences based on the TD-error.
    """
    def __init__(self, buffer_size, observation_space, action_space, alpha=0.6, beta=0.4, beta_increment=1e-4, eps=1e-6, rank_based=True, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, observation_space, action_space, **kwargs)
        self.alpha = alpha  # Controls how much prioritization is used (0 is uniform, 1 is full prioritization)
        self.beta = beta  # Controls how much importance-sampling correction is used
        self.beta_increment = beta_increment  # Rate at which beta increases to 1
        self.eps = eps
        self.rank_based = rank_based
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)  # Initialize priorities to zero

    def add(self, *args, **kwargs):
        """
        Add a transition to the replay buffer with the maximum priority.

        Args:
            *args: Variable length argument list to be passed to the parent class's add method.
            **kwargs: Arbitrary keyword arguments to be passed to the parent class's add method.

        The priority of the new transition is set to the maximum priority currently in the buffer.
        If the buffer is empty, the priority is set to 1.0.
        """
        # Add a transition with the maximum priority
        max_priority = self.priorities.max() if self.size() > 0 else 1.0
        super(PrioritizedReplayBuffer, self).add(*args, **kwargs)
        self.priorities[self.pos - 1] = max_priority  # Set priority for the new transition

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the prioritized replay buffer.
        Args:
            batch_size (int): The number of experiences to sample.
        Returns:
            tuple: A tuple containing:
                - batch (list): A batch of sampled experiences.
                - weights (torch.Tensor): The importance-sampling weights for the sampled experiences.
        Notes:
            - The sampling probabilities are calculated based on the priorities of the experiences.
            - The importance-sampling weights are used to correct for the bias introduced by prioritized sampling.
            - The beta parameter is incremented over time to anneal the importance-sampling weights.
        """
        # Calculate sampling probabilities
        #if self.size == self.buffer_size:
        #    probs = self.priorities ** self.alpha
        #else:
        #    probs = self.priorities[:self.pos] ** self.alpha
        probs = self.priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample a batch of indices based on probabilities
        indices = np.random.choice(len(probs), size=batch_size, p=probs)
        batch = super(PrioritizedReplayBuffer, self)._get_samples(indices)
        
        # Calculate importance-sampling weights
        total = len(probs)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        weights = torch.tensor(weights)

        # Update beta toward 1 over time
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, indices, weights

    def update_priorities(self, indices, new_priorities):
        """
        Update the priorities of the replay buffer based on the TD errors.

        Args:
            indices (np.ndarray): The indices of the transitions to update.
            new_priorities (np.ndarray): The new priorities corresponding to the indices.
        """
        self.priorities[indices] = new_priorities.squeeze() + self.eps