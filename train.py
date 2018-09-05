from user_simulator import UserSimulator
from emc import EMC
from Agents.rule_based import RuleBasedAgent
from Agents.dqn import DQNAgent

MAX_DIAL_PER_CONV = 10

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

# Outer Loop
total_dialogue = 0
while True:
    # User sim picks goal and reset agenda
    user_sim.reset(goal_list)
    # Inner loop (by conversation)
    conv_dialogue = 0
    succ = False
    while conv_dialogue < MAX_DIAL_PER_CONV:
        # User sim picks action
        user_frame = user_sim.output_action(None)
        # Infuse error
        user_frame = emc_0.infuse_error(user_frame)
        # Sends to agent
        agent_frame = dqn_agent.output_action(user_frame)
        # agent updates policy (based on some condition)
        dqn_agent.train()

        total_dialogue += 1
        conv_dialogue += 1

        # If passes done
        if agent_frame is ...:
            succ = user_sim.check_conv_success()
            break

    conv_dialogue = 0
