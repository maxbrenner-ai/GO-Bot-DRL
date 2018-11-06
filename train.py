from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle
import json
import math
from utils import remove_empty_slots


# Load constants json into dict
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
WARMUP_MEM = run_dict['warmup_mem']
NUM_EP_TRAIN = run_dict['num_ep_run']
TRAIN_FREQ = run_dict['train_freq']
MAX_ROUND_NUM = run_dict['max_round_num']
SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']

# Load movie DB
database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

# Clean DB
remove_empty_slots(database)

# Load movie dict
db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

# Load goal File
user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

# Init. Objects
user_sim = UserSimulator(user_goals, constants, database)
emc = ErrorModelController(db_dict, constants)
state_tracker = StateTracker(database, constants)
dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


# Warmup loop
def warmup_run():
    print('Warmup Started...')
    ep = 0
    total_step = 0
    done_warmup = False
    while not done_warmup:
        ep_reset()
        ep += 1
        ep_step = 0
        ep_reward = 0
        done = False
        while not done:
            # Get state tracker state
            state = state_tracker.get_state()
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = dqn_agent.get_action(state, use_rule=True)
            # Update state tracker with the agent's action
            round_num = state_tracker.update_state_agent(agent_action)
            # User takes action given agent action
            user_action, reward, done, success = user_sim.step(agent_action, round_num)
            ep_reward += reward
            if not done:
                # Infuse error into semantic frame level of user action
                emc.infuse_error(user_action)
                # Update state tracker with user action
            state_tracker.update_state_user(user_action)
            # Add memory
            next_state = state_tracker.get_state(done)
            dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

            ep_step += 1
            total_step += 1

            if total_step == WARMUP_MEM or dqn_agent.is_memory_full():
                done_warmup = True
                done = True
    print('...Warmup Ended')


# Training Loop
def train_run():
    print('Training Started...')
    ep = 0
    period_rew_total = 0
    period_min_reward = math.inf
    period_max_reward = -math.inf
    period_success_total = 0
    success_rate_best = 0.0
    while ep < NUM_EP_TRAIN:
        ep_reset()
        # Inner loop (by conversation)
        ep += 1
        ep_reward = 0
        done = False
        while not done:
            state = state_tracker.get_state()
            agent_action_index, agent_action = dqn_agent.get_action(state)
            round_num = state_tracker.update_state_agent(agent_action)
            user_action, reward, done, success = user_sim.step(agent_action, round_num)
            ep_reward += reward
            period_rew_total += reward
            if not done:
                emc.infuse_error(user_action)
            state_tracker.update_state_user(user_action)
            next_state = state_tracker.get_state(done)
            dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

        period_min_reward = min(period_min_reward, ep_reward)
        period_max_reward = max(period_max_reward, ep_reward)

        if success:
            period_success_total += 1

        if ep % TRAIN_FREQ == 0:
            # Check success rate
            success_rate = period_success_total / TRAIN_FREQ
            avg_reward = period_rew_total / TRAIN_FREQ
            # Flush
            if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
                dqn_agent.empty_memory()
            # Update current best success rate
            if success_rate > success_rate_best:
                print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {} Min Rew: {} Max Rew: {}'.format(ep, success_rate, avg_reward, period_min_reward, period_max_reward))
                success_rate_best = success_rate
                dqn_agent.save_weights()
            period_success_total = 0
            period_rew_total = 0
            period_min_reward = math.inf
            period_max_reward = -math.inf
            # Copy
            dqn_agent.copy()
            # Train
            dqn_agent.train()
    print('...Training Ended')


# User takes first action
def ep_reset():
    # First reset the state tracker
    state_tracker.reset()
    # Then pick an init user action
    user_action = user_sim.reset()
    # Infuse with error
    user_error_action = emc.infuse_error(user_action)
    # And update state tracker
    state_tracker.update_state_user(user_error_action)
    # Finally, reset agent
    dqn_agent.reset()


def main():
    warmup_run()
    train_run()


if __name__ == "__main__":
    main()