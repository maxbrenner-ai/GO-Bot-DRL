# Special slot values (Not for use, just reference)
'PLACEHOLDER'  # For informs
'UNK'  # For requests
'anything'  # = to I DO NOT CARE, means any value works for the slot with this value
'match available'  # For when the agent intent to inform a match is found, and the ST does find at least one
'no match available'  # Used for same above and Used for inform slots needed to be filled by db_helper but there was no match with the current informs

#######################################
# Usersim Config
#######################################
# Used in EMC for intent error
usersim_intents = ['inform', 'request', 'thanks', 'reject']

# Used for the end goal of the usersim, eg ticket, it must be added to req slots of the usergoal
usersim_default_key = 'ticket'

# Required to be in the first action in inform slots of the usersim if they exist in goal inform slots
usersim_required_init_inform_keys = ['moviename']

#######################################
# Agent Config
#######################################

# Possible inform and request slots for the agent
agent_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids', usersim_default_key]
agent_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids']

# Possible actions for agent
agent_actions = [
    {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
    {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
]
for slot in agent_inform_slots:
    agent_actions.append({'intent': 'inform', 'inform_slots': {slot: 'PLACEHOLDER'}, 'request_slots': {}})
for slot in agent_request_slots:
    agent_actions.append({'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}})

# Rule-based policy request list
rule_requests = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

# These are possible inform slot keys that cannot be used to query
no_query_keys = ['numberofpeople', usersim_default_key]

#######################################
# Global config
#######################################

# These are used for both constraint check AND success check in usersim
FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1


# All possible intents (for one-hot conversion in ST.get_state())
# done for agent if it is closing, or usersim if round num past max round num
all_intents = ['inform', 'request', 'done', 'match_found', 'thanks', 'reject']

# All possible slots (for one-hot conversion in ST.get_state())
all_slots = ['actor', 'actress', 'city', 'critic_rating', 'date', 'description', 'distanceconstraints',
             'genre', 'greeting', 'implicit_value', 'movie_series', 'moviename', 'mpaa_rating',
             'numberofpeople', 'numberofkids', 'other', 'price', 'seating', 'starttime', 'state',
             'theater', 'theater_chain', 'video_format', 'zip', 'result', 'ticket', 'mc_list']