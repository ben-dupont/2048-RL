This project aims at implementing deep reinforcement learning to train an agent that can play the 2048 puzzle game. It was a pretext to learn, have a fun, and maybe more.

# Project details

- Experiments are launched from the 2048.ipynb and run locally on a MacBook using MPS.
- The enviornment is implemented in gym2048.py as a sub-class of OpenAI Gym.
- Two RL algorithms have been implemented and tested: Temporal Difference Learning (TDL) and Proximal Policy Optimization (PPO), and are respectively implemented in TDL.py and PPO.py
- The CNN network used for the project are inspired from K. Matsuzaki paper: [Developing Value Networks for Game 2048
with Reinforcement Learning](https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_pdf)
- Multiple techniques have been experimented for the TDL algorithm, including data augmentation based on the symmetry of the board, Prioritized Experience Replay (PER) and 2-ply expectimax.

# Key results

TDL worked very well, as already proved in the litterature. 

![results](https://github.com/user-attachments/assets/22ebe083-febe-4333-91b5-4ebbcdc11b1a)

Best results were obtained with TDL:
- using the symmetry of the board for data augmentation,
- training in batches of 1024 moves (no prioritized experience replay),
- with the 2-ply expectimax policy at inference time (not during training).

![Symmetry only](https://github.com/user-attachments/assets/d3876590-04d4-4139-bd40-cca725ca982d)

PPO also works reasonably well, but training is slower. After 10,000 games, it reaches the 1024 tile at best, while TDL easily reaches 2048 and occasionally 4096. 

![PPO 2nd test](https://github.com/user-attachments/assets/0665c152-e897-4b33-afc0-61704ac8cd1c)
