# Goal-Oriented Chatbot trained with Deep Reinforcement Learning

Based off of the code repo [TC-Bot](https://github.com/MiuLab/TC-Bot) and paper [End-to-End Task-Completion Neural Dialogue Systems](http://aclweb.org/anthology/I17-1074). This repo is a simplified version of TC-Bot, it performs at a similar level of accuracy (although it cannot be compared directly as success conditions have been altered).

## Details

This shows how to train a simple DQN agent with Deep Reinforcement Learning as a goal-oriented chatbot using a simple user simulator. 

## Dependencies
- Python >= 3.5
- Keras >= 2.24 (Earlier versions probably work)
- numpy

## How to Run
You can train an agent from scratch by entering ```python train.py``` in console or just running train.py in an IDE. 

In constants.json you can change hyperparameters including "save_weights_file_path" and "load_weights_file_path" (both relative paths) to save and load weights respectively. Weights for both target and behavior keras models are saved everytime the current success rate is at a new high. 

You can also test an agent with ```python test.py```. But make sure to load weights in with "load_weights_file_path" in constants.json set to a relative path with beh and tar weights. 

All the constants are pretty self explanatory other than "vanilla" under agent which means DQN (true) or Double DQN (false). Defualt is vanilla DQN. 

## Test (or Train) with an Actual User
You can test the agent by inputing your own actions as the user (instead of using a user sim) by setting "usersim" under run in constants.json to false. You input an action and a success indicator every step of an episode/conversation in console. The format is: intent/inform slots/request slots.

Example Action Inputs:
- request/moviename: MIB, date: friday/time, cost, reviews
- inform/moviename: zooptopia/
- request//time
- done//

In addition the console will ask for an indicator on whether the agent succeeded yet (other than after the initial action input of an episode). Allowed inputs are -1 for loss, 0 for no outcome yet, 1 for success. 

## My Data
Used hyperparameters from constants.json.

Table of episodes (every 2000 out of 40000) by max success rate of a period/train frequency (every 100 episodes) up to that episode:

![Data Table](https://github.com/maxbren/GO-Bot_DRL/blob/master/assets/data_table.PNG)
