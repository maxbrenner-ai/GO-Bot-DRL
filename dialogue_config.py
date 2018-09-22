# Special slot values (Not for use, just reference)
'PLACEHOLDER'  # For informs
'UNK'  # For requests
'anything'  # = to I DO NOT CARE, means any value works for the slot with this value
'match available'  # For when the agent intent to inform a match is found, and the ST does find at least one
'no match available'  # Used for same above and Used for inform slots needed to be filled by db_helper but there was no match with the current informs


#######################################
# Global config
#######################################

# All possible intents (for one-hot conversion in ST.get_state())
# Todo: so i am gonna see if i can make match_found an intent instead of inform slot
all_intents = ['inform', 'request', 'done', 'match_found', 'accept', 'reject']

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
    {'intent': 'match_found', 'inform_slots': {'match': 'PLACEHOLDER'}, 'request_slots': {}}  # To say that a match was found
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
