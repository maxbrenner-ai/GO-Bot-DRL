from user_simulator import UserSimulator
from emc import EMC
from dqn_agent import DQNAgent
from dqn_agent_2 import DQNAgent as DQNAgent_2
from state_tracker import StateTracker
import pickle, csv
import json
import math


# Load Constants Json into Dict
CONSTANTS_FILE_PATH = 'constants.json'
with open(CONSTANTS_FILE_PATH) as f:
    constants = json.load(f)

# Load File Path Constants
file_path_dict = constants['db_file_paths']
DATABASE_FILE_PATH = file_path_dict['database']
DICT_FILE_PATH = file_path_dict['dict']
USER_GOALS_FILE_PATH = file_path_dict['user_goals']

# Load Run Constants
run_dict = constants['run']
NUM_EP_TEST = run_dict['num_ep_run']
MAX_ROUND_NUM = run_dict['max_round_num']

# Load Movie DB
database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

# Todo: Figure out how to augment the db itself instead of having to do this
# This removes all items with values of '' (ie values of empty string)
for id in list(database.keys()):
    for key in list(database[id].keys()):
        if database[id][key] == '':
            database[id].pop(key)

# Load Movie Dict
db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')
# Load Goal File
user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

# Init. Objects
user_sim = UserSimulator(user_goals, constants, database)
emc_0 = EMC(db_dict, constants)
state_tracker = StateTracker(database, constants)
dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


def test_run():
    print('Testing Started...')
    ep = 0
    while ep < NUM_EP_TEST:
        ep_reset()
        # Inner loop (by conversation)
        ep += 1
        ep_reward = 0
        done = False
        while not done:
            # Get state tracker state
            state = state_tracker.get_state()
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = dqn_agent.get_action(state)
            # Update state tracker with the agent's action
            round_num = state_tracker.update_state_agent(agent_action)
            # User sim. takes action given agent action
            user_action, reward, done, succ = user_sim.step(agent_action, round_num)
            ep_reward += reward
            if not done:
                # Infuse error into semantic frame level user sim. action
                emc_0.infuse_error(user_action)
            # Update state tracker with user sim. action
            state_tracker.update_state_user(user_action)
        print('Episode: {} Succ.: {} Reward: {}'.format(ep, succ, ep_reward))
    print('...Testing Ended')


# User sim takes first action
def ep_reset():
    # First reset the state tracker
    state_tracker.reset()
    # Then pick an init user action
    user_action = user_sim.reset()
    # Infuse with error
    user_error_action = emc_0.infuse_error(user_action)
    # And update state tracker
    state_tracker.update_state_user(user_error_action)
    # Finally, reset agent
    dqn_agent.reset()


def main():
    test_run()


if __name__ == "__main__":
    main()