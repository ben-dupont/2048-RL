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

    row_1, col_1 = np.random.randint(0, 4, size=2)
    row_2, col_2 = np.random.randint(0, 4, size=2)
    while row_1 == row_2 and col_1 == col_2:
        row_2, col_2 = np.random.randint(0, 4, size=2)

    tile_layer = random.randint(1,self.observation_space.shape[0]) if jump else 1

    self.state[tile_layer, row_1, col_1] = 1
    self.state[0, row_2, col_2] = 1

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
    observation = np.array(self.encode(new_mat), dtype=np.float32)
    return observation, float(reward)

  def step(self, action):
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
    observation, reward = self.afterstate(self.state, action)
    zero_idx = np.argwhere(observation[0] == 1)
    
    if zero_idx.size == 0:
      terminated = True
    else:
      tile_layer = 2 if random.random() < 0.1 else 1 # 90% chance of 2, 10% chance of 4
      rand_idx = random.choice(zero_idx)
      observation[0, rand_idx[0], rand_idx[1]] = 0
      observation[tile_layer, rand_idx[0], rand_idx[1]] = 1
      self.state = observation
      terminated = not any(self.allowed_actions())

    return self.state, reward, terminated, False, {}

  def allowed_actions(self):
    """
    Determine the allowed actions in the current state of the 2048 game.

    This method checks which moves (left, right, up, down) are possible from the current state
    by simulating each move and comparing the resulting matrices to the current state matrix to 
    see if they are different.

    Returns:
      list of bool: A list of four boolean values indicating whether each move (left, right, up, down)
              is allowed. True means the move is allowed, and False means it is not.
    """
    mat = self.decode(self.state)
    mat_0, _ = self._swipe(mat, 'left')
    mat_1, _ = self._swipe(mat, 'right')
    mat_2, _ = self._swipe(mat, 'up')
    mat_3, _ = self._swipe(mat, 'down')

    return [(mat != mat_0).any(), (mat != mat_1).any(), (mat != mat_2).any(), (mat != mat_3).any()]

  def playOneGame(self, model=None, verbose=True):
    """
    Plays one game of 2048 with random actions until the game terminates.

    Parameters:
    model (optional): The model to be evaluated. If None, random actions will be taken.
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
      if model:
        with torch.no_grad():
          action, _ = model.predict(observation=self.state)
        action = action.item()
      else:
        action = random.randrange(0,4)
      observation, reward, terminated, _, _ = self.step(action)
      cum_reward += reward

    board = self.decode(observation)
    max_tile = np.max(board)

    if verbose:
      print("Final state:")
      print(board)
      print("Max tile: %i" % max_tile)
      print("Score: %i" % cum_reward)
      print("Number of moves: %i" % cpt)
    return max_tile, cpt

  def evaluate(self, model=None, n_games=1000):
    """
    Evaluate the performance of a given model by playing a specified number of games.

    Parameters:
    model (optional): The model to be evaluated. If None, random actions will be taken.
    n_games (int): The number of games to be played for evaluation. Default is 1000.

    Returns:
    tuple: Two numpy arrays:
      - max_tiles: An array containing the maximum tile value achieved in each game.
      - n_steps: An array containing the number of steps taken in each game.
    """
    max_tiles = np.array([])
    n_steps = np.array([])

    for k in tqdm(range(n_games)):
      max_tile, cpt = self.playOneGame(model=model, verbose=False)
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
      print("\n Playing one random game \n")
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
    y = np.array([0] * 4)

    idx = 0

    for j in range(4):
      if x[j] != 0:
        y[idx] = x[j]
        idx += 1

    return y

  def _merge(self, x):
    """
    Merges the tiles in a row or column for the 2048 game.

    This function takes a list of integers representing the tiles in a row or column
    and merges adjacent tiles that have the same value. The merged tile will have the
    value of the sum of the two tiles, and the second tile will be set to zero. The
    function also keeps track of the total value of the merged tiles.

    Args:
      x (list of int): A list of integers representing the tiles in a row or column.

    Returns:
      tuple: A tuple containing:
        - y (list of int): The list of integers after merging the tiles.
        - merge_cpt (int): The total value of the merged tiles.
    """
    y = x

    merge_cpt = 0

    for j in range(3):
      if x[j] == x[j+1]:
        y[j] = x[j] + x[j+1]
        x[j+1] = 0
        merge_cpt += y[j]
      else:
        y[j] = x[j]

    return y, merge_cpt

  def _swipeLeft(self, x):
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