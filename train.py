from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json, math
from utils import remove_empty_slots
from user import User


if __name__ == "__main__":
    # Can provide constants file path in args OR run it as is and change 'CONSTANTS_FILE_PATH' below
    # 1) In terminal: python train.py --constants_path "constants.json"
    # 2) Run this file as is
    parser = argparse.ArgumentParser()
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)

    # Load constants json into dict
    CONSTANTS_FILE_PATH = 'constants.json'
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
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
    # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

    # Clean DB
    remove_empty_slots(database)

    # Load movie dict
    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

    # Load goal File
    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    # Init. Objects
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)
    else:
        user = User(constants)
    emc = ErrorModelController(db_dict, constants)
    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


def warmup_run():
    """
    Runs the warmup stage of training which is used to fill the agents memory.

    The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
    Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.

    """

    print('Warmup Started...')
    ep = 0
    total_step = 0
    done_warmup = False
    while not done_warmup:
        episode_reset()
        ep += 1
        ep_reward = 0
        done = False
        # Get initial state from state tracker
        state = state_tracker.get_state()
        while not done:
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = dqn_agent.get_action(state, use_rule=True)
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
            # Add memory
            next_state = state_tracker.get_state(done)
            dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)
            # Update state
            state = next_state

            total_step += 1

        if total_step == WARMUP_MEM or dqn_agent.is_memory_full():
            break
    print('...Warmup Ended')


def train_run():
    """
    Runs the loop that trains the agent.

    Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
    TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.

    """

    print('Training Started...')
    episode = 0
    period_rew_total = 0
    period_min_reward = math.inf
    period_max_reward = -math.inf
    period_success_total = 0
    success_rate_best = 0.0
    while episode < NUM_EP_TRAIN:
        episode_reset()
        # Inner loop (by conversation)
        episode += 1
        episode_reward = 0
        done = False
        state = state_tracker.get_state()
        while not done:
            agent_action_index, agent_action = dqn_agent.get_action(state)
            round_num = state_tracker.update_state_agent(agent_action)
            user_action, reward, done, success = user.step(agent_action, round_num)
            episode_reward += reward
            period_rew_total += reward
            if not done:
                emc.infuse_error(user_action)
            state_tracker.update_state_user(user_action)
            next_state = state_tracker.get_state(done)
            dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)
            state = next_state

        period_min_reward = min(period_min_reward, episode_reward)
        period_max_reward = max(period_max_reward, episode_reward)

        if success:
            period_success_total += 1

        if episode % TRAIN_FREQ == 0:
            # Check success rate
            success_rate = period_success_total / TRAIN_FREQ
            avg_reward = period_rew_total / TRAIN_FREQ
            # Flush
            if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
                dqn_agent.empty_memory()
            # Update current best success rate
            if success_rate > success_rate_best:
                print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {} Min Rew: {} Max Rew: {}'
                      .format(episode, success_rate, avg_reward, period_min_reward, period_max_reward))
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


def episode_reset():
    """
    Resets the episode/conversation in the warmup and training loops.

    Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.

    """

    # First reset the state tracker
    state_tracker.reset()
    # Then pick an init user action
    user_action = user.reset()
    # Infuse with error
    emc.infuse_error(user_action)
    # And update state tracker
    state_tracker.update_state_user(user_action)
    # Finally, reset agent
    dqn_agent.reset()


warmup_run()
train_run()
