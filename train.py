from user_simulator import UserSimulator
from emc import EMC
from Agents.rule_based import RuleBasedAgent
from Agents.dqn import DQNAgent
import constants as C

MAX_EP_LENGTH = 10

# Load corpus
# Make rule based agent
rule_based_agent = RuleBasedAgent()
# Generate possible user goals
goal_list = ...
# Init user sim
user_sim = UserSimulator()
# Init error model controller
emc_0 = EMC(level=1, type=1, error_amount=0.05)
# Init or load agent
dqn_agent = DQNAgent()
state_tracker = StateTracker()

# Warm-Up loop
def warmup():
    total_step = 0
    done_warmup = False
    while not done_warmup:
        # User sim picks goal and reset agenda
        user_sim.reset(goal_list)
        ep_step = 0
        user_frame = ...
        while ep_step < C['max_ep_length']:
            # Agent chooses action given state/user_frame
            agent_frame = dqn_agent.return_action(user_frame)
            # Step user sim
            next_user_frame, reward, done = user_sim.step(agent_frame)
            # Add to memory
            ...

            ep_step += 1
            total_step += 1

            if total_step == C['warmup_mem']:
                done_warmup = True

            # If passes done
            if agent_frame is ... or done_warmup:
                succ = user_sim.check_conv_success()
                break


# Training Loop
def train():
    ep = 0
    period_succ_total = 0
    succ_rate_best = 0.0
    while ep < C['num_ep_train']:
        # User sim picks goal and reset agenda
        user_sim.reset(goal_list)
        # Inner loop (by conversation)
        ep += 1
        ep_step = 0
        while ep_step < C['max_ep_length']:

            # Agent takes action given state tracker's representation of dialogue
            agent_action = dqn_agent.get_action(state_tracker.get_state())
            # Update state tracker with the agent's action
            state_tracker.update_state_agent(agent_action)
            # User sim. takes action given agent action
            user_action, reward, done, succ = user_sim.get_action(agent_action)
            # Infuse error into semantic frame level user sim. action
            user_error_action = emc_0.infuse_error(user_action)
            # Update state tracker with user sim. action
            state_tracker.update_state_user(user_error_action)

            ep_step += 1

            if done:
                if succ:
                    period_succ_total += 1
                break

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


def test():
    ep = 0
    while ep < C['num_ep_test']:
        user_sim.reset(goal_list)
        ep += 1
        ep_step = 0
        while ep_step < C['max_ep_length']:
            # Agent chooses action given state/user_frame
            agent_frame = dqn_agent.return_action(user_frame)
            # Step user sim
            next_user_frame, reward, done = user_sim.step(agent_frame)

            ep_step += 1

            # If passes done
            if agent_frame is ...:
                succ = user_sim.check_conv_success()
                break

warmup()
train()
test()