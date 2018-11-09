from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle
import json
from user import User
from utils import remove_empty_slots

# Load Constants json into a dict
CONSTANTS_FILE_PATH = 'constants.json'
with open(CONSTANTS_FILE_PATH) as f:
    constants = json.load(f)

# Load file path constants
file_path_dict = constants['db_file_paths']
DATABASE_FILE_PATH = file_path_dict['database']
DICT_FILE_PATH = file_path_dict['dict']
USER_GOALS_FILE_PATH = file_path_dict['user_goals']

# Load run constants
run_dict = constants['run']
USE_USERSIM = run_dict['usersim']
NUM_EP_TEST = run_dict['num_ep_run']
MAX_ROUND_NUM = run_dict['max_round_num']

# Load movie DB
database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

# Clean DB
remove_empty_slots(database)

# Load movie dict
db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

# Load goal file
user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

# Init. Objects
if USE_USERSIM:
    user = UserSimulator(user_goals, constants, database)
else:
    user = User(constants)
emc = ErrorModelController(db_dict, constants)
state_tracker = StateTracker(database, constants)
dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


def test_run():
    """
    Runs the loop that tests the agent.

    Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent. Terminates when the episode
    reaches NUM_EP_TEST.

    """

    print('Testing Started...')
    episode = 0
    while episode < NUM_EP_TEST:
        episode_reset()
        episode += 1
        ep_reward = 0
        done = False
        while not done:
            # Get state tracker state
            state = state_tracker.get_state()
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = dqn_agent.get_action(state)
            # Update state tracker with the agent's action
            round_num = state_tracker.update_state_agent(agent_action)
            # User takes action given agent action
            user_action, reward, done, success = user.step(agent_action, round_num)
            ep_reward += reward
            if not done:
                # Infuse error into semantic frame level of user action
                emc.infuse_error(user_action)
            # Update state tracker with user action
            state_tracker.update_state_user(user_action)
        print('Episode: {} Success: {} Reward: {}'.format(episode, success, ep_reward))
    print('...Testing Ended')


def episode_reset():
    """Resets the episode/conversation in the testing loop."""

    # First reset the state tracker
    state_tracker.reset()
    # Then pick an init user action
    user_action = user.reset()
    # Infuse with error
    user_error_action = emc.infuse_error(user_action)
    # And update state tracker
    state_tracker.update_state_user(user_error_action)
    # Finally, reset agent
    dqn_agent.reset()


if __name__ == "__main__":
    test_run()
