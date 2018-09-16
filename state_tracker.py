from db_query import DBQuery
import numpy as np


class StateTracker:
    def __init__(self, movie_database):
        self.db_helper = DBQuery(movie_database)
        self.reset()

    def reset(self):
        self.current_informs = {}
        self.history = []  # Is a list of the dialogues (dict) by the agent and user so far in the conversation
        self.turn = 1

    def get_state(self):
        user_action = self.history[-1]
        kb_results_dict = self.db_helper.database_results_for_agent(self.current_informs)
        last_agent_action = self.history[-2] if len(self.history) > 1 else None

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in self.current_informs:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if last_agent_action:
            agent_act_rep[0, self.act_set[last_agent_action['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if last_agent_action:
            for slot in last_agent_action['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if last_agent_action:
            for slot in last_agent_action['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        turn_rep = np.zeros((1, 1)) + self.turn / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################

        # Todo: I'm interested to see if including this increases performance

        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, self.turn] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
            kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return final_representation

    def update_state_agent(self, agent_action):
        inform_slots = self.db_helper.fill_inform_slots(agent_action['inform_slots'], self.current_informs)
        agent_action[inform_slots] = inform_slots
        for slot in agent_action['inform_slots'].keys():
            self.current_informs[slot] = agent_action['inform_slots'][slot]  # add into inform_slots
        self.history.append(agent_action)
        self.history[-1].update({'turn': self.turn, 'speaker': 'Agent'})
        self.turn += 1
        return agent_action  # I may not tech. have to return it, i think it updates anyway, but this is easier to read

    def update_state_user(self, user_action):
        for slot in user_action['inform_slots'].keys():
            self.current_informs[slot] = user_action['inform_slots'][slot]
        self.history.append(user_action)
        # Todo: check if this still changes the user_action object, dont really want it to
        self.history[-1].update({'turn': self.turn, 'speaker': 'User'})
        self.turn += 1
