import numpy as np
import random
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import torch

class gym2048(gym.Env):
  """
  A custom Gym environment for playing the 2048 game. This environment
  simulates the 2048 game mechanics, including the grid, actions, and rewards.

  Attributes:
      cumulative_reward (float): The cumulative reward earned during the game.
      observation_space (gym.spaces.Box): The space representing valid observations.
      action_space (gym.spaces.Discrete): The space representing valid actions (left, right, up, down).
      act_dict (dict): A dictionary mapping action indices to their corresponding directions.
  """
  def __init__(self, max_tile=8192):
    """
    Initialize the environment.

    Parameters:
        max_tile (int): The maximum tile value in the game, must be a power of 2 (default is 8192).
    """
    # custom class variable used to display the reward earned
    self.cumulative_reward = 0

    # observation space (valid ranges for observations in the state)
    self.observation_space = spaces.Box(shape=(int(np.log2(max_tile))+1, 4, 4), low=0, high=1, dtype=np.float32)

    # valid actions:
    self.action_space = spaces.Discrete(4)
    self.act_dict = {0: 'left', 1:'right', 2:'up', 3:'down'}

  def reset(self, seed=None, jump=False):
    """
    Reset the environment to an initial state.

    Parameters:
        seed (int): Random seed for reproducibility.
        jump (bool): If True, initializes the state with a random higher tile.

    Returns:
        tuple: The initial state and an empty info dictionary.
    """
    super().reset(seed=seed)
    random.seed(seed)

    # set the initial state to a flattened 4x4 grid with two randomly Twos
    self.state = np.zeros((self.observation_space.shape[0],4,4), dtype=np.float32)
    self.state[0, :, :] = 1

    row_1, col_1 = np.random.randint(0, 4, size=2)
    row_2, col_2 = np.random.randint(0, 4, size=2)
    while row_1 == row_2 and col_1 == col_2:
        row_2, col_2 = np.random.randint(0, 4, size=2)

    tile_layer = random.randint(1,self.observation_space.shape[0]-1) if jump else 1

    self.state[tile_layer, row_1, col_1] = 1
    self.state[0, row_1, col_1] = 0
    self.state[1, row_2, col_2] = 1
    self.state[0, row_2, col_2] = 0

    return self.state, {}

  def encode(self, mat):
    """
    Encode a 4x4 matrix into a 16x4x4 tensor, representing the presence of different powers of 2.

    Parameters:
        mat (numpy.ndarray): A 4x4 matrix representing the current state.

    Returns:
        numpy.ndarray: A tensor of shape (16, 4, 4) encoding the grid state.
    """
    tensor = np.zeros((self.observation_space.shape[0],4,4), dtype=np.float32)
    for i in range(4):
      for j in range(4):
        if mat[i,j] > 0:
          k = int(np.log2(mat[i,j]))
          tensor[k,i,j] = 1
        else: 
          tensor[0,i,j] = 1
    return tensor

  def decode(self, tensor):
    """
    Decode the 16x4x4 tensor back into a 4x4 matrix.

    Parameters:
        tensor (numpy.ndarray): A tensor of shape (16, 4, 4) encoding the grid.

    Returns:
        numpy.ndarray: The decoded 4x4 matrix representing the grid state.
    """
    powers_of_two = 2 ** np.arange(1,self.observation_space.shape[0], dtype=np.float32)[:, None, None]  # Shape: (16, 1, 1)
    mat = np.sum(tensor[1:,:,:] * powers_of_two, axis=0)  # Shape: (4, 4)
    return mat

  def afterstate(self, state, action):
    """
    Calculate the resulting afterstate and reward after taking an action, but before adding a random tile.

    Parameters:
        state (numpy.ndarray): The current state (encoded).
        action (int): The action to take (0: left, 1: right, 2: up, 3: down).

    Returns:
        tuple: The resulting state (encoded) before adding a random tile and the reward.
    """
    mat = self.decode(state)
    new_mat, reward = self._swipe(mat, self.act_dict[action])
    observation = self.encode(new_mat)
    return observation, float(reward)

  def step(self, action, state=None):
    """
    Execute one step in the environment based on the given action.

    Parameters:
    action (int): The action to be taken.

    Returns:
    tuple: A tuple containing:
      - state (object): The new state of the environment after the action.
      - reward (float): The reward obtained from taking the action.
      - terminated (bool): Whether the game has ended.
      - truncated (bool): Always False, included for compatibility with OpenAI Gym.
      - info (dict): Additional information, currently empty.
    """
    state = state if state is not None else self.state
    observation, reward = self.afterstate(state, action)
    zero_idx = np.argwhere(observation[0] == 1)
    
    if zero_idx.size == 0: # No empty cells left
      terminated = True
    else:
      # Decide the new tile's value: 90% chance for 2 (layer 1), 10% for 4 (layer 2)
      tile_layer = 2 if random.random() < 0.1 else 1

      # Choose a random empty cell to place the new tile
      rand_idx = random.choice(zero_idx)

      # Update the state with the new tile
      observation[0, rand_idx[0], rand_idx[1]] = 0
      observation[tile_layer, rand_idx[0], rand_idx[1]] = 1

      self.state = observation
      terminated = not any(self.allowed_actions())

    return observation, reward, terminated, False, {}

  def allowed_actions(self, state=None):
    """
    Determine the allowed actions in the current state of the 2048 game.

    This method checks which moves (left, right, up, down) are possible from the current state
    by simulating each move and comparing the resulting matrices to the current state matrix to 
    see if they are different.

    Args:
      state (any): The current state of the game, encoded as a (16,4,4) tensor.

    Returns:
      list of bool: A list of four boolean values indicating whether each move (left, right, up, down)
              is allowed. True means the move is allowed, and False means it is not.
    """
    state = state if state is not None else self.state
    mat = self.decode(state)
    mat_0, _ = self._swipe(mat, 'left')
    mat_1, _ = self._swipe(mat, 'right')
    mat_2, _ = self._swipe(mat, 'up')
    mat_3, _ = self._swipe(mat, 'down')

    return [(mat != mat_0).any(), (mat != mat_1).any(), (mat != mat_2).any(), (mat != mat_3).any()]

  def all_possible_next_states(self, afterstate):
    """
    Generate all possible next states from the given afterstate.

    This function takes an afterstate (a state after a move has been made) and 
    generates all possible next states by placing a '2' or a '4' in each empty 
    cell (represented by a '1' in the afterstate). The probabilities of placing 
    a '2' or a '4' are 0.9 and 0.1 respectively, distributed evenly among all 
    empty cells.

    Args:
      afterstate (np.ndarray): The current state of the game after a move has 
                   been made. It is a 3D numpy array where the 
                   first dimension represents the different 
                   possible values (empty, 2, 4) and the other 
                   two dimensions represent the game grid.

    Returns:
      list of tuples: A list of tuples where each tuple contains a probability 
              and the corresponding next state. The next state is a 
              3D numpy array similar to the input afterstate.
    """
    possible_next_states = []
    zero_idx = np.argwhere(afterstate[0] == 1)
    n_zero = zero_idx.shape[0]
    if n_zero > 0:
      prob_2 = 0.9 / n_zero
      prob_4 = 0.1 / n_zero
      for idx in zero_idx:
        next_state_2 = afterstate.copy()
        next_state_2[0, idx[0], idx[1]] = 0
        next_state_2[1, idx[0], idx[1]] = 1
        possible_next_states.append((prob_2, next_state_2))
        next_state_4 = afterstate.copy()
        next_state_4[0, idx[0], idx[1]] = 0
        next_state_4[2, idx[0], idx[1]] = 1
        possible_next_states.append((prob_4, next_state_4))
    return possible_next_states

  def playOneGame(self, policy=None, verbose=True, **kwargs):
    """
    Plays one game of 2048 with random actions until the game terminates.

    Parameters:
    model (optional): The model to be evaluated. If None, random actions will be taken.
    policy (int): The policy to be used for action selection. 1 = greedy, 2+ = expectimax with the specified depth. Default is 1.
    verbose (bool): If True, prints the final state of the game board, the cumulative reward, and the number of steps taken. Default is True.

    Returns:
      None

    Prints:
      - The final state of the game board.
      - The cumulative reward obtained during the game.
      - The number of steps taken to reach the terminal state.
    """
    terminated = False
    self.reset()
    cpt = 0
    cum_reward = 0

    while terminated == False:
      cpt += 1
      if policy:
        with torch.no_grad():
          action, _ = policy(self.state, **kwargs)
        action = action.item()
      else:
        action = random.randrange(0,4)
      observation, reward, terminated, _, _ = self.step(action)
      cum_reward += reward

      if verbose and cpt % 100 == 0:
        print("Step: %i" % cpt)
        print(self.decode(observation).astype(int))

    board = self.decode(observation)
    max_tile = np.max(board)

    if verbose:
      print("Final state:")
      print(board.astype(int))
      print("Max tile: %i" % max_tile)
      print("Score: %i" % cum_reward)
      print("Number of moves: %i" % cpt)
    return max_tile, cpt

  def evaluate(self, policy=None, n_games=1000, **kwargs):
    """
    Evaluate the performance of a given model by playing a specified number of games.

    Parameters:
    model (optional): The model to be evaluated. If None, random actions will be taken.
    policy (int): The policy to be used for action selection. 1 = greedy, 2+ = expectimax with the specified depth. Default is 1.
    n_games (int): The number of games to be played for evaluation. Default is 1000.

    Returns:
    tuple: Two numpy arrays:
      - max_tiles: An array containing the maximum tile value achieved in each game.
      - n_steps: An array containing the number of steps taken in each game.
    """
    max_tiles = np.array([])
    n_steps = np.array([])

    for k in tqdm(range(n_games)):
      max_tile, cpt = self.playOneGame(policy=policy, verbose=False, **kwargs)
      max_tiles = np.append(max_tiles, max_tile)
      n_steps = np.append(n_steps, cpt)

    return max_tiles, n_steps

  def check(self):
      """
      Perform a series of checks on the 2048 game environment and its functionalities.
      This method performs the following checks:
      1. Environment check using `check_env`.
      2. Swipe left operation on various test arrays and compares the results.
      3. Swipe operations (left, right, up, down) on a matrix and compares the results.
      4. Swipe down operation on a specific test array and compares the result.
      5. Encoding and decoding of the game state and verifies the correctness.
      6. Afterstate generation for different actions and verifies the correctness.
      7. Plays one random game to ensure the game logic is functioning.
      The method prints the results of each check to the console.
      """
      
      # Check environment
      check_env(self, warn=True)

      # Check SwipeLeft
      print("\n Check SwipeLeft \n")
      a = np.array([0,0,2,2])
      aa = np.array([4,0,0,0])
      a_cpt = 4

      b = np.array([0,2,2,2])
      bb = np.array([4,2,0,0])
      b_cpt = 4

      c = np.array([2,0,2,0])
      cc = np.array([4,0,0,0])
      c_cpt = 4

      d = np.array([0,2,0,0])
      dd = np.array([2,0,0,0])
      d_cpt = 0

      e = np.array([2,2,2,2])
      ee = np.array([4,4,0,0])
      e_cpt = 8

      sa, sa_cpt = self._swipeLeft(a)
      sb, sb_cpt = self._swipeLeft(b)
      sc, sc_cpt = self._swipeLeft(c)
      sd, sd_cpt = self._swipeLeft(d)
      se, se_cpt = self._swipeLeft(e)
      print(aa == sa, a_cpt == sa_cpt)
      print(bb == sb, b_cpt == sb_cpt)
      print(cc == sc, c_cpt == sc_cpt)
      print(dd == sd, d_cpt == sd_cpt)
      print(ee == se, e_cpt == se_cpt)

      # Check Swipe
      print("\n Check SwipeLeft \n")
      m = np.array([a,b,c,d])
      mm_left = np.array([aa, bb, cc, dd])
      m_left_cpt = 12

      mm_right = np.array([[0,0,0,4],[0,0,2,4],[0,0,0,4],[0,0,0,2]])
      m_right_cpt = 12

      mm_up = np.array([[2,4,4,4],[0,0,2,0],[0,0,0,0],[0,0,0,0]])
      m_up_cpt = 12

      mm_down = np.array([[0,0,0,0],[0,0,0,0],[0,0,2,0],[2,4,4,4]])
      m_down_cpt = 12

      print(m)
      print(mm_left)
      print(mm_right)
      print(mm_up)
      print(mm_down)

      print(self._swipe(m,'left')[0] == mm_left, self._swipe(m,'left')[1] == m_left_cpt)
      print(self._swipe(m,'right')[0] == mm_right, self._swipe(m,'right')[1] == m_right_cpt)
      print(self._swipe(m,'up')[0] == mm_up, self._swipe(m,'up')[1] == m_up_cpt)
      print(self._swipe(m,'down')[0] == mm_down, self._swipe(m,'down')[1] == m_down_cpt)

      test = np.array([0.,  0.,  8., 64., 0.,  4., 32.,  4., 0.,  0.,  0.,  8., 2.,  0.,  0., 4.]).reshape((4,4))
      result_down = np.array([0., 0., 0., 64., 0., 0., 0., 4., 0., 0., 8., 8., 2., 4., 32., 4.]).reshape((4,4))
      print(self._swipe(test,'down')[0]==result_down)

      # Check encode / decode
      print("\n Check encode / decode \n")
      env = gym2048()
      print(m)
      print(env.encode(m))
      print(env.decode(env.encode(m)))
      print(env.decode(env.encode(m)) == m)

      # Check afterstate
      print("\n Check Afterstate \n")
      env = gym2048()
      state = env.encode(m)
      print(env.decode(env.afterstate(state, 0)[0]) == mm_left)
      print(env.decode(env.afterstate(state, 1)[0]) == mm_right)
      print(env.decode(env.afterstate(state, 2)[0]) == mm_up)
      print(env.decode(env.afterstate(state, 3)[0]) == mm_down)

      # Play one game
      print("\n Playing one random game\n")
      self.playOneGame()

  def _get_obs(self):
    """
    Returns the current state of the environment.
    
    Returns:
      The current state of the environment.
    """
    return self.state

  def _compress(self, x):
    """
    Compresses the input array by shifting all non-zero elements to the left.

    Parameters:
    x (numpy.ndarray): A 1D numpy array of length 4 containing integers.

    Returns:
    numpy.ndarray: A 1D numpy array of length 4 with all non-zero elements of the input array shifted to the left and zeros filled in the remaining positions.
    """
    result = np.zeros_like(x)
    non_zero_count = 0
    for value in x:
        if value != 0:
            result[non_zero_count] = value
            non_zero_count += 1
    return result

  def _merge(self, x):
      """
      Merges the tiles in a row or column for the 2048 game

      Parameters:
      x (numpy.ndarray): A 1D numpy array of length 4 containing integers representing the tiles.

      Returns:
      tuple:
          - numpy.ndarray: A 1D numpy array of length 4 after merging the tiles.
          - int: The total value of the merged tiles.
      """
      x = np.array(x)  # Ensure input is a NumPy array
      merge_cpt = 0

      # Iterate over the array and merge adjacent equal values
      for j in range(len(x) - 1):
          if x[j] != 0 and x[j] == x[j + 1]:  # Only merge non-zero tiles
              x[j] *= 2
              x[j + 1] = 0
              merge_cpt += x[j]

      return x, merge_cpt

  def _swipeLeft(self, row):
    """
    Swipe a row to the left according to the rules of the 2048 game.

    This function takes a row of the 2048 game board, removes zeros, and combines
    adjacent tiles with the same value by summing them. The resulting row is then
    padded with zeros to maintain the original length of 4.

    Parameters:
    row (numpy.ndarray): A 1D numpy array of length 4 representing a row of the 2048 game board.

    Returns:
    tuple: A tuple containing the updated row after the left swipe and the count of merges performed.
    """
    row = row[row != 0] # remove zeros
    l = len(row)
    merge_cpt = 0

    if l == 0:
        return np.zeros(4, dtype=np.float32), 0
    elif l == 1:
        return np.array([row[0], 0, 0, 0]), 0
    elif l == 2:
        if row[0] == row[1]:
            #return np.concatenate((row[0]*2, np.zeros(3, dtype=np.float32))), row[0]*2
            return np.array([row[0]*2, 0, 0, 0]), row[0]*2
        else:
            #return np.concatenate((row, np.zeros(2, dtype=np.float32))), 0
            return np.array([row[0], row[1], 0, 0]), 0
    elif l == 3:
        if row[0] == row[1]:
            #return np.concatenate((row[0]*2, row[2], np.zeros(2, dtype=np.float32))), row[0]*2
            return np.array([row[0]*2, row[2], 0, 0]), row[0]*2
        elif row[1] == row[2]:
            #return np.concatenate((row[0], row[1]*2, np.zeros(2, dtype=np.float32))), row[1]*2
            return np.array([row[0], row[1]*2, 0, 0]), row[1]*2
        else:
            #return np.concatenate((row, np.zeros(1, dtype=np.float32))), 0
            return np.array([row[0], row[1], row[2], 0]), 0
    else:
        if row[0] == row[1]:
            if row[2] == row[3]:
                return np.array([row[0]*2, row[2]*2, 0, 0]), row[0]*2 + row[2]*2
            else:
                return np.array([row[0]*2, row[2], row[3], 0]), row[0]*2
        elif row[1] == row[2]:
            return np.array([row[0], row[1]*2, row[3], 0]), row[1]*2
        elif row[2] == row[3]:
            return np.array([row[0], row[1], row[2]*2, 0]), row[2]*2
        else:
            return row, 0

  def _swipeLeft_old(self, x):
    """
    Simulates a left swipe action on the game board.

    Args:
      x (list): A list representing a row of the game board.

    Returns:
      tuple: A tuple containing the updated row after the left swipe and the count of merges performed.
    """
    y = self._compress(x)
    y, merge_cpt = self._merge(y)
    y = self._compress(y)
    return y, merge_cpt

  def _swipe(self, mat, dir):
    """
    Perform a swipe operation on the given matrix in the specified direction.

    Parameters:
    mat (np.ndarray): The 4x4 matrix representing the game board.
    dir (str): The direction to swipe. Can be 'right', 'up', or 'down'.

    Returns:
    tuple: A tuple containing the new matrix after the swipe and the total number of merges made during the swipe.

    Notes:
    - The 'left' direction is implicitly handled by the _swipeLeft method.
    - The matrix is temporarily transformed to handle swipes in different directions using numpy operations.
    """
    t_mat = mat.copy()
    merge_cpt = 0
    if dir == 'right':
        t_mat = t_mat[:, ::-1]
    elif dir == 'up':
        t_mat = np.transpose(t_mat)
    elif dir == 'down':
        t_mat = np.transpose(t_mat)[:, ::-1]

    for i in range(4):
        t_mat[i], merge_cpt_loc = self._swipeLeft(t_mat[i])
        merge_cpt += merge_cpt_loc

    if dir == 'right':
        t_mat = t_mat[:, ::-1]
    elif dir == 'up':
        t_mat = np.transpose(t_mat)
    elif dir == 'down':
        t_mat = np.transpose(t_mat[:, ::-1])

    return t_mat, merge_cpt