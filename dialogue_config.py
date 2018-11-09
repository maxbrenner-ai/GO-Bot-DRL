# Special slot values (for reference)
'PLACEHOLDER'  # For informs
'UNK'  # For requests
'anything'  # means any value works for the slot with this value
'no match available'  # When the intent of the agent is match_found yet no db match fits current constraints

#######################################
# Usersim Config
#######################################
# Used in EMC for intent error (and in user)
usersim_intents = ['inform', 'request', 'thanks', 'reject', 'done']

# The goal of the agent is to inform a match for this key
usersim_default_key = 'ticket'

# Required to be in the first action in inform slots of the usersim if they exist in the goal inform slots
usersim_required_init_inform_keys = ['moviename']

#######################################
# Agent Config
#######################################

# Possible inform and request slots for the agent
agent_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating',
                      'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor',
                      'description', 'other', 'numberofkids', usersim_default_key]
agent_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip',
                       'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price',
                       'actor', 'description', 'other', 'numberofkids']

# Possible actions for agent
agent_actions = [
    {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
    {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
]
for slot in agent_inform_slots:
    # Must use intent match found to inform this, but still have to keep in agent inform slots
    if slot == usersim_default_key:
        continue
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
all_intents = ['inform', 'request', 'done', 'match_found', 'thanks', 'reject']

# All possible slots (for one-hot conversion in ST.get_state())
all_slots = ['actor', 'actress', 'city', 'critic_rating', 'date', 'description', 'distanceconstraints',
             'genre', 'greeting', 'implicit_value', 'movie_series', 'moviename', 'mpaa_rating',
             'numberofpeople', 'numberofkids', 'other', 'price', 'seating', 'starttime', 'state',
             'theater', 'theater_chain', 'video_format', 'zip', 'result', usersim_default_key, 'mc_list']
