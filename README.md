# Bananna colector

<center>
	<img src="https://github.com/eduardodisanti/drl_banana_collector/blob/master/media/banana_collector.gif" alt="agent in action" width="480"/>
</center>

## Introduction

This project shows how to train the Banana colector environment in the Udacity modified (from Unity's Bannana-colector) environment.
The goal of the agent is to collect as many yellow bananas while avoiding blue bananas. The environment grants positive rewards for yellow bananas and negative for blue ones.
This project trains the agent to play the game using [ray-based](https://en.wikipedia.org/wiki/Ray_tracing_(graphics)) perception of objects arround the agent forward direction. Check [Report file](https://github.com/eduardodisanti/drl_banana_collector/blob/master/report/Report.pdf) for technical specifications, algorithm, hyperparameters and architecture of the neural network.

## Files description

|  File | Description | Location |
|-------|-------------|----------|
| Navigation.ipynb  | Notebook that trains the model and track progress | / |
| play_banana.ipynb  | A notebook to see the agent in action and assess it | / |
| dqn_agent.py  | Defines the DQN agent to solve the environment. | /code |
| model.py  | Defines the Neural Network that the agents uses to generalize actions. | /code |
| auxs.py  | Some helper functions like moving average. | /code |
| train_agent_headless.py  | Contains code for training the agent that can be run standalone | /code |
| Report.pdf  | Technical paper and training report | /report |
| banana_raytracing_eds.pt | Saved weights of the agent neural network | /model
| banana_play.m4v | A video showing the trained agend | /media

## Instructions 
*Refer to "Setting up the environment" section to download the environment for this project in case you don't have the Unity environment installed.*

### Training the agent

- Training in a Jupyter notebook (recommended), 
	Open the Navigation notebook and run all cells <br/>
	During the training a graph will be displayed, updated at the end of every episode.<br/>
  <img src="https://github.com/eduardodisanti/drl_banana_collector/blob/master/report/training.png" width="180"/><br/>
	At the end of the training a final figure with appear showing the rewards, average, moving average and goal<br/>
	<center>
		<img src="https://github.com/eduardodisanti/drl_banana_collector/blob/master/report/training2.png" alt="training report" width="180"/>
	</center>
	
- Training from the command line,
	Execute *cd code*<br/>
	Run the train_agent_headless.py using the command *python train_agent_headless.py*<br/>
	During the training the progress of the training will be written to the console. <br/>

The training will stop when the agent reach an average of +13 on last 100 scenarios. (Will happen around 330 episodes) 

### Assesing the agent

In a jupyter notebook open play_banana.ipynb and run all cells.<br>
At the end of 100 trails of 2000 steps each, a report will be shown containing the returns achieved by the agent.
<img src="https://github.com/eduardodisanti/drl_banana_collector/blob/master/report/play_scores.png" alt="drawing" width="240"/><br/>

## Apendices

### Agent characteristics
*Please check [Report file](https://github.com/eduardodisanti/drl_banana_collector/blob/master/report/Report.pdf) for a better understanding of the algorithm*
#### Algorithm
The agent uses a  [Deep Q-Learning algorithm](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
The neural network that estimate the action-value function has following architecture:

|  Layer | Neurons Size  | Type | Activation | Comment |
|--------|-------|------|------------|---------|
|Input  |    37 | | | according to the space state dimension | 
|Hidden 1  |  32 | Linear | ReLU |
|Hidden 2  |  32 | Linear | ReLU |
|Output  |  4 | Linear | ReLU | One for each action

The  optimization algorithm used was **Adam**
Chossen learning rate **5e-4**
##### Hyperparameters
-   Discount factor, $\gamma$ 0.99
-   Soft-update ratio, $\tau$     0.001
-   Network update interval
    -  Local every **4** time steps.
    -  Target every **4** time steps.
-   Replay buffer size  **64**
-   Minibatch size **64**
-   &epsilon; configuration and decay rate for **&epsilon;-greedy policy**
    -   Start          : 1
    -   Minumin.  : 0.01
    -   Decay rate: 0.995

 With the above hyperparameters, the average score of the last 100 episodes reached 13.03 after 321 episodes.
The agent performed actions acording to this table:

| Action | Frequency |
| ------ | --------- |
| 0 | 57234 |
| 1 | 25189 |
| 2 | 9690 |
| 3 | 4187 |
<br/>
And the reward history was:
<table>
<tr>
<td><img src="https://github.com/eduardodisanti/drl_banana_collector/blob/master/report/training.png" width="480"/></td>
<td><img src="https://github.com/eduardodisanti/drl_banana_collector/blob/master/report/training2.png" width="520"/></td>
<tr>
</table>

#### Future work
One behaiviour observed during training and playing, wich is coherent with, the high bias of DQN and the action frequencies shown, is the poor capacity for the agent to turn right, turning left was better learned.<br>
The agent perform well, but in the future may worth to:
 - increase the number of neurons in the hidden layers
 - Use prioritized experience play
 - Use dueling DQN

#### Running the play_banana
In the 7th cell the parameters can be adjusted, by default it runs 100 times with 2000 steps.
An example of results :
*Start game...*<br/>
. . . . <br/>
Total Reward: 13.0<br/>
Total Reward: 11.0<br/>
Total Reward: 15.0<br/>
Total Reward: 15.0<br/>
Total Reward: 20.0<br/>
Total Reward: 19.0<br/>
. . . . <br/>
Total Reward: 18.0<br/>
Total Reward: 19.0<br/>
Total Reward: 10.0<br/>
Total Reward: 14.0<br/>
. . . . <br/>
*Game Over*<br/>
<img src="https://github.com/eduardodisanti/drl_banana_collector/blob/master/report/play_scores.png" alt="drawing" width="480"/><br/>

#### Setting up the environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.

2. Then, place the files and folders of this repository in the `p1_navigation/` folder in the DRLND GitHub repository, you can overwrite any file in conflict.  Next, open `Navigation.ipynb` to train the agent.

