import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class gym2048(gym.Env):
  def __init__(self, log_reward=False):
    self.n_grid = 4*4
    # custom class variable used to display the reward earned
    self.cumulative_reward = 0
    self.log_reward = log_reward

    # observation space (valid ranges for observations in the state)
    self.observation_space = spaces.Box(shape=(16, 4, 4), low=0, high=1, dtype=np.float32)

    # valid actions:
    #   0 = left
    #   1 = right
    #   2 = up
    #   3 = down
    # spaces.Discrete(4) is a shortcut for defining the actions 0-3
    self.action_space = spaces.Discrete(4)
    self.act_dict = {0: 'left', 1:'right', 2:'up', 3:'down'}

  def _get_obs(self):
    return self.state

  def encode(self, mat):
    # Input: 4x4 matrice
    # Output: 16x4x4 tensor
    # see https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_pdf/-char/en
    tensor = np.zeros((16,4,4), dtype=np.float32)
    for i in range(4):
      for j in range(4):
        if mat[i,j] > 0:
          k = int(np.log2(mat[i,j]))
          tensor[k,i,j] = 1
    return tensor

  def decode(self, tensor):
    # Input: 16x4x4 tensor
    # Output: 4x4 matrice
    # see https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_pdf/-char/en
    mat = np.zeros((4,4), dtype=np.float32)
    for k in range(16):
      for i in range(4):
        for j in range(4):
          if tensor[k,i,j] == 1:
            mat[i,j] += 2**k
    return mat

  def reset(self, seed=None):
    super().reset(seed=seed)
    random.seed(seed)

    # set the initial state to a flattened 4x4 grid with two randomly Twos
    vec = np.zeros(self.n_grid, dtype=np.float32)
    index_1 = random.randrange(0, self.n_grid)
    index_2 = random.randrange(0, self.n_grid)
    while index_2 == index_1:
      index_2 = random.randrange(0, self.n_grid)
    vec[index_1] = 2
    vec[index_2] = 2

    self.state = np.array(self.encode(vec.reshape((4,4))), dtype=np.float32)

    return self.state, {}

  def afterstate(self, state, action):
    reward = 0.0

    mat = self.decode(state)

    #zero_count = np.sum(mat == 0)
    #grid_value = np.sum(mat)
    #max_tile = np.max(mat)

    act = self.act_dict[action]
    new_mat, merge_cpt = swipe(mat, act)

    #new_zero_count = np.sum(new_mat == 0)
    #new_grid_value = np.sum(new_mat)
    new_max_tile = np.max(new_mat)
    zero_idx = np.where(new_mat == 0)
    #reach_2048 = np.where(new_mat == 2048)[0].size

    if len(zero_idx[0]) == 0:
      #terminated = (mat == swipe(mat,0)).all() and (mat == swipe(mat,1)).all() and (mat == swipe(mat,2)).all() and (mat == swipe(mat,3)).all()
      terminated = True
      reward += -100
    elif new_max_tile == 2048:
      terminated = True
      reward += 1000
    else:
      terminated = False

      if merge_cpt > 0:
        if self.log_reward:
          reward += np.log2(merge_cpt)
        else:
          reward += merge_cpt
      else:
        reward += -1

    observation = np.array(self.encode(new_mat), dtype=np.float32)

    return observation, reward, terminated, False, {}

  def step(self, action):
    observation, reward, terminated, _, _ = self.afterstate(self.state, action)

    new_mat = self.decode(observation)

    if not terminated:
        zero_idx = np.where(new_mat == 0)

        rand_new_tile = random.randrange(0,10)
        if rand_new_tile == 0:
            new_tile = 4
        else:
            new_tile = 2
        rand_idx = random.randrange(0,len(zero_idx[0]))
        new_mat[zero_idx[0][rand_idx], zero_idx[1][rand_idx]] = new_tile

    self.state = np.array(self.encode(new_mat), dtype=np.float32)

    return self.state, reward, terminated, False, {}

  def allowed_actions(self):
    a_size = int(np.sqrt(self.n_grid))
    mat = self.state.reshape((a_size, a_size))
    mat_0 = swipe(mat, 'left')
    mat_1 = swipe(mat, 'right')
    mat_2 = swipe(mat, 'up')
    mat_3 = swipe(mat, 'down')

    return [(mat != mat_0).any(), (mat != mat_1).any(), (mat != mat_2).any(), (mat != mat_3).any()]

def compress(x):
  y = np.array([0] * 4)

  idx = 0

  for j in range(4):
    if x[j] != 0:
      y[idx] = x[j]
      idx += 1

  return y

def merge(x):
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

def swipeLeft(x):
  y = compress(x)
  y, merge_cpt = merge(y)
  y = compress(y)
  return y, merge_cpt

def swipe(mat, dir):
  t_mat = mat.copy()
  merge_cpt = 0
  match dir:
    case 'right':
      t_mat = t_mat[:, ::-1]
    case 'up':
      t_mat = np.transpose(t_mat)
    case 'down':
      t_mat = np.transpose(t_mat)[:, ::-1]

  for i in range(4):
    t_mat[i], merge_cpt_loc = swipeLeft(t_mat[i])
    merge_cpt += merge_cpt_loc

  match dir:
    case 'right':
      t_mat = t_mat[:, ::-1]
    case 'up':
      t_mat = np.transpose(t_mat)
    case 'down':
      t_mat = np.transpose(t_mat[:, ::-1])

  return t_mat, merge_cpt

def check_game():
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

    sa, sa_cpt = swipeLeft(a)
    sb, sb_cpt = swipeLeft(b)
    sc, sc_cpt = swipeLeft(c)
    sd, sd_cpt = swipeLeft(d)
    se, se_cpt = swipeLeft(e)
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

    print(swipe(m,'left')[0] == mm_left, swipe(m,'left')[1] == m_left_cpt)
    print(swipe(m,'right')[0] == mm_right, swipe(m,'right')[1] == m_right_cpt)
    print(swipe(m,'up')[0] == mm_up, swipe(m,'up')[1] == m_up_cpt)
    print(swipe(m,'down')[0] == mm_down, swipe(m,'down')[1] == m_down_cpt)

    test = np.array([0.,  0.,  8., 64., 0.,  4., 32.,  4., 0.,  0.,  0.,  8., 2.,  0.,  0., 4.]).reshape((4,4))
    result_down = np.array([0., 0., 0., 64., 0., 0., 0., 4., 0., 0., 8., 8., 2., 4., 32., 4.]).reshape((4,4))
    print(swipe(test,'down')[0]==result_down)

    # Check afterstate
    print("\n Check Afterstate \n")
    env = gym2048()
    state = env.encode(m)
    print(env.decode(env.afterstate(state, 0)[0]) == mm_left)
    print(env.decode(env.afterstate(state, 1)[0]) == mm_right)
    print(env.decode(env.afterstate(state, 2)[0]) == mm_up)
    print(env.decode(env.afterstate(state, 3)[0]) == mm_down)