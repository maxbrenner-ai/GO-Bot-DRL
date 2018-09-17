# Special slot values (Not for use, just reference)
'PLACEHOLDER'  # For informs
'UNK'  # For requests

#######################################
# Global config
#######################################

# All possible intents (for one-hot conversion in ST.get_state())
all_intents = ['inform', 'request', 'done', 'accept', 'reject']

# All possible slots (for one-hot conversion in ST.get_state())
all_slots = [...]

#######################################
# Agent Config
#######################################

# Possible inform and request slots for the agent
agent_inform_slots = []
agent_request_slots = []

# Possible actions for agent
agent_actions = [
    {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
]
for slot in agent_inform_slots:
    agent_actions.append({'intent': 'inform', 'inform_slots': {slot: 'PLACEHOLDER'}, 'request_slots': {}})
for slot in agent_request_slots:
    agent_actions.append({'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}})

# Rule-based policy request list
rule_requests = [...]

#######################################
# Usersim Config
#######################################

# Possible inform and request slot values for usersim
usersim_inform_slots = [...]
usersim_request_slots = [...]
