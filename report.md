# Project Report

Here is a report of the method used to solve the tennis multi agent unity environment. The environment involves two agents controlling rackets to bounce a ball over a net. The observation is continuous with 
24 values corresponding to the position and velocity of the ball and racket. The  action space is continuous being a 
2 dimensional vector corresponding to the movement of the racket towards the net, and jumping. An agent receives a 
reward of 0.1 for hitting the ball over the net and -0.01 for letting the ball hit the ground or go out of bounds.

The task is solved when the average score of 100 episodes greater than or equal to 0.5. The score of each episode is considered as the 
maximum of the undiscounted returns from the agents.

The PPO Implementation from the previous project on [continuous control](https://github.com/ldfrancis/continuous_control_PPO) was used to solve this environment.

## PPO Agent
A PPO agent has a policy which it tries to optimize using experiences from the environment using that policy, and with 
the help of importance sampling, old experience can be reused for this optimization. The PPO implementation uses a 
policy model, a critic model, and the clipped surrogate loss function. The PPO agent implementation can be found in [agent.py](ippo/agent.py)

**Policy model**

The policy model is an MLP that outputs a distribution over actions from which an agent can sample from. Its input is 
the observation from the environment. It has 3 layers with the first and last hidden layers having 64 and 32 units 
respectively. Each hidden unit is followed by a rectified linear unit activation. For the output, a normal distribution
is constructed using a mean and standard deviation. The mean is obtained as an output of a layer tha takes as input, 
the output of the last hidden layer. This layer uses a tanh activation to keep the mean in the range [-1,1]. For 
obtaining standard deviation, a layer was made to, instead, output the log standard deviation and the exp was taken 
to obtain the standard deviation. This is convenient to use with a neural network as a setting is impossed for the 
minimum and maximumn log standard deviation, the network outputs a value in the range [-1,1] and this formular is used 
for the log standard deviation;

MIN_LOG_STD + (tanh(log_std)+1)*(MAX_LOG_STD-MIN_LOG_STD)/2

The settings for the MIN_LOG_STD and MAX_LOG_STD are found in the [config.py](config.py) file.

**Critic model**

The model for the critic is a simple MLP with 3 layers. The output layer is with one unit with no activation and the input to the model is the observation (33 dim vector) from the environment. Each of the 2 hidden layers are followed by a relu activation.

The policy and critic models implementation can be found in [model.py](ippo/model.py)

**Clipped Surrogate loss**

The clipped surrogate loss was implemented. First the policy ratio is computed (current_policy/old_policy). A clipped version of this ratio is also computed using the CLIP_EPSILON parameter from [config.py](config.py) (clip(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON)). With these ratios, two surrogate objectives are calculated and the minimum is taken as the objective. The surrogate loss is obtained by negating the sum of this objective and the entropy.


## Training parameters

Traning was done using an adam optimizer with a learning rate of 5e-5 for both the policy and critic with batch size of 64. Collected experiences were repeatedly used to train the agent for a number of times equal to EPOCHS. The parameters for training can be configured in the [config.py](config.py)

Below is a description of the parameters found in [config.py](config.py).

Name | data type | Use
--- | --- | ---
BATCH_SIZE | int | used when sampling experiences from a trajectory for training
GAMMA | float | discount factor
LAMBDA | float | used for gae calculation 
POLICY_HIDDEN_DIM | list | hidden layers for policy model
CRITIC_HIDDEN_DIM | list | hidden layers for critic model
MAX_LOG_STD | float | for calculating the standard deviation for the action distribution
MAX_LOG_STD | float | for calculating the standard deviation for the action distribution
ENTROPY_WEIGHT | float | weight for entropy in surrogate loss
EPOCHS | int | number of times to train using the collected experiences before proceeding
POLICY_LR | float | learning rate for the policy optimizer
CRITIC_LR | float | learning rate for the critic optimizer
CLIP_EPSILON | float | for calculating the clipped ratio of current policy and old policy


## Results

After training for 191 episodes, A mean score across all agents (averaged over the last 100 episodes) of 30.01 was attained.
below is an image of the training progress, plots of scores and average scores attained at each episode.

![plots/score_plot.png](plots/score_plot.png)

The result achieved with the PPO agent is outlined below;

Episodes | average score (last 100 episodes) | evaluation score
--- | --- | ---
191 | 30.01 | 35.09

## Possible Improvements

For improvements over the current results, other settings of the hyper-parameters can be tried out. Also, other algorithms like, DDPG, A2C, can be explored.