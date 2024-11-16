import torch
import torch.nn as nn
from gymnasium import spaces

class CNN22(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for processing observations in a reinforcement learning environment.
    Args:
        observation_space (spaces.Box): The observation space of the environment, which defines the shape of the input data.
    Attributes:
        cnn (nn.Sequential): The sequential container of convolutional layers and activation functions.
        fc (nn.Sequential): The sequential container of fully connected layers and activation functions.
    Methods:
        forward(observations: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the CNN and fully connected layers.
        truncated_normal_init(m):
            Applies truncated normal initialization to the weights of linear and convolutional layers.
    """
    def __init__(self, observation_space: spaces.Box):
        super(CNN22, self).__init__()

        # Extract observation dimensions
        n_input_channels = observation_space.shape[0]

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=2, stride=1),  # 1st conv layer
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=1),  # 2nd conv layer
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the output size of the CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        # Fully connected layers after CNN
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the neural network.

        Args:
            observations (torch.Tensor): The input tensor containing observations.

        Returns:
            torch.Tensor: The output tensor after passing through the CNN and fully connected layers.
        """
        # Forward pass through CNN and FC layers
        return self.fc(self.cnn(observations))
    
    def truncated_normal_init(m):
        """
        Initialize the weights of a given layer using truncated normal distribution.

        This function applies truncated normal initialization to the weights of 
        layers of type `nn.Linear` or `nn.Conv2d`. The weights are initialized 
        with a mean of 0 and a standard deviation of 0.1, truncated to the range 
        [-0.2, 0.2]. If the layer has biases, they are initialized to zero.

        Args:
            m (nn.Module): The layer to initialize.

        Returns:
            None
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # Apply truncated normal initialization
            nn.init.trunc_normal_(m.weight, mean=0, std=0.1, a=-2 * 0.1, b=2 * 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to zero (or another value if needed)