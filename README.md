# multi_agent_tennis
*A multi agent tennis environment to be solved by an RL agent, for udacity deep reinforcement learning nanodegree*

Last project for DRLND.

The environment involves two agents controlling rackets to bounce a ball over a net. The observation is continuous with 
24 values corresponding to the position and velocity of the ball and racket. The  action space is continuous being a 
2 dimensional vector corresponding to the movement of the racket towards the net, and jumping. An agent receives a 
reward of 0.1 for hitting the ball over the net and -0.01 for letting the ball hit the ground or go out of bounds.

The task is solved when the average score of 100 episodes greater than or equal to 0.5. The score of each episode is considered as the 
maximum of the undiscounted returns from the agents.

## Getting Started
This project uses python 3 and some of its packages. To get started, first, install anaconda/miniconda  and then create the conda environment using;

```conda create --name=drlndmarl python=3.6```

```source activate drlndmarl```

Clone the repository

```git clone https://github.com/ldfrancis/multi_agent_tennis.git```

Change the current working directory to the projects base folder

```cd multi_agent_tennis```

Then proceed to installing the required packages by running

```pip install -r requirements.txt```

Having installed all the required packages, the unity environment files can then be downloaded and placed in the 
tennis_env folder. Below are links to download the unity environments for the popular operating systems;

[linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip) <br/>
[mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip) <br/>
[windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/Tennis_Windows_x86.zip) <br/>
[windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Tennis/Tennis_Windows_x86_64.zip) <br/>

Also, the PLATFORM variable in the config.py file has to be modified accordingly.

Now, all is set for the experiments

## Instructions
To run the experiment use the commands below:

Train an agent in the environment;

```python main.py train```

Evaluate a trained agent

```python main.py test```

This would use the default configs specified in ```config.py```. The file config.py contains variables whose values are
necessary to configure the environment, the agent, and the experiment. Below is a sample setting for the variables in 
config.py
```
...
ENV_PATH = f"./tennis_env/{ENV_FILE}"
NUM_OBS = 33
NUM_ACT = 4
TARGET_SCORE = 30

# ddpg agent
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
ACTOR_HIDDEN_DIM = [256, 128]
CRITIC_HIDDEN_DIM = [256, 128]
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
```

The experiment would continue running for several episodes till the agents achieve a max score of 0.5 averaged over the 
last 100 episodes.

## Report
A report containing the results can be found [here](report.md)
