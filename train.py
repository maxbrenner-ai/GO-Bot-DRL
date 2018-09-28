from user_simulator import UserSimulator
from emc import EMC
from dqn_agent import DQNAgent
import constants as C
from state_tracker import StateTracker

MAX_EP_LENGTH = 10

# Load corpus
# Generate possible user goals
goal_list = ...
# Init user sim
user_sim = UserSimulator(goal_list)
# Init error model controller
emc_0 = EMC(level=1, type=1, error_amount=0.05)
state_tracker = StateTracker()
# Init or load agent
dqn_agent = DQNAgent(state_tracker.get_state_size())

# Warm-Up loop
def warmup():
    total_step = 0
    done_warmup = False
    while not done_warmup:
        ep_reset()
        ep_step = 0
        done = False
        while not done:
            # Get state tracker state
            state = state_tracker.get_state()
            # Agent takes action given state tracker's representation of dialogue
            agent_action = dqn_agent.get_action(state)
            # Update state tracker with the agent's action
            agent_action, round_num = state_tracker.update_state_agent(agent_action)
            # User sim. takes action given agent action
            user_action, reward, done, succ = user_sim.step(agent_action, round_num)
            if not done:
                # Infuse error into semantic frame level user sim. action
                user_error_action = emc_0.infuse_error(user_action)
                # Update state tracker with user sim. action
                state_tracker.update_state_user(user_error_action)
            # Add memory
            dqn_agent.add_experience(state, agent_action, reward, state_tracker.get_state(), done)

            ep_step += 1
            total_step += 1

            if total_step == C['warmup_mem']:
                done_warmup = True
                done = True


# Training Loop
def train():
    ep = 0
    period_succ_total = 0
    succ_rate_best = 0.0
    while ep < C['num_ep_train']:
        ep_reset()
        # Inner loop (by conversation)
        ep += 1
        done = False
        while not done:
            # Get state tracker state
            state = state_tracker.get_state()
            # Agent takes action given state tracker's representation of dialogue
            agent_action = dqn_agent.get_action(state)
            # Update state tracker with the agent's action
            # Todo: Note a round counts as an action by a user then agent (incremented in update_state_agent in ST)
            agent_action, round_num = state_tracker.update_state_agent(agent_action)
            # User sim. takes action given agent action
            user_action, reward, done, succ = user_sim.step(agent_action, round_num)
            if not done:
                # Infuse error into semantic frame level user sim. action
                user_error_action = emc_0.infuse_error(user_action)
                # Update state tracker with user sim. action
                state_tracker.update_state_user(user_error_action)
             # Add memory
            dqn_agent.add_experience(state, agent_action, reward, state_tracker.get_state(), done)

        if succ:
            period_succ_total += 1

        if ep % C['val_freq'] == 0:
            # Check succ rate
            succ_rate = period_succ_total / C['val_freq']
            if succ_rate >= succ_rate_best and succ_rate >= C['success_rate_threshold']:
                # Flush
                dqn_agent.empty_memory()
                succ_rate_best = succ_rate
            period_succ_total = 0
            # Copy
            dqn_agent.copy()
            # Train
            dqn_agent.train()


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


def test():
    ep = 0
    while ep < C['num_ep_test']:
        ep_reset()
        ep += 1
        ep_step = 0
        done = False
        while not done:
            # Get state tracker state
            state = state_tracker.get_state()
            # Agent takes action given state tracker's representation of dialogue
            agent_action = dqn_agent.get_action(state)
            # Update state tracker with the agent's action
            agent_action, round_num = state_tracker.update_state_agent(agent_action)
            # User sim. takes action given agent action
            user_action, reward, done, succ = user_sim.step(agent_action, round_num)
            if not done:
                # Infuse error into semantic frame level user sim. action
                user_error_action = emc_0.infuse_error(user_action)
                # Update state tracker with user sim. action
                state_tracker.update_state_user(user_error_action)

            ep_step += 1


warmup()
train()
test()