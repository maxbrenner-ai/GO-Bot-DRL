from db_query import DBQuery


class StateTracker:
    def __init__(self, movie_database):
        self.db_helper = DBQuery(movie_database)
        self.reset()

    def reset(self):
        self.current_informs = {}
        self.history = []  # Is a list of the dialogues (dict) by the agent and user so far in the conversation
        self.turn = 1

    # Unlike in TC-Bot, i am going to return the state in the correct format for the agent's input
    def get_state(self):
        ...

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
