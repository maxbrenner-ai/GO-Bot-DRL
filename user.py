from dialogue_config import FAIL, SUCCESS, usersim_intents, all_slots
from utils import reward_function


class User:
    """Connects a real user to the conversation through the console."""

    def __init__(self, constants):
        """
        The constructor for User.

        Parameters:
            constants (dict): Loaded constants as dict
        """
        self.max_round = constants['run']['max_round_num']

    def reset(self):
        """
        Reset the user.

        Returns:
            dict: The user response
        """

        return self._return_response()

    def _return_response(self):
        """
        Asks user in console for response then receives a response as input.

        Format must be like this: request/moviename: room, date: friday/starttime, city, theater
        or inform/moviename: zootopia/
        or request//starttime
        or done//
        intents, informs keys and values, and request keys and values cannot contain / , :

        Returns:
            dict: The response of the user
        """

        response = {'intent': '', 'inform_slots': {}, 'request_slots': {}}
        while True:
            input_string = input('Response: ')
            chunks = input_string.split('/')

            intent_correct = True
            if chunks[0] not in usersim_intents:
                intent_correct = False
            response['intent'] = chunks[0]

            informs_correct = True
            if len(chunks[1]) > 0:
                informs_items_list = chunks[1].split(', ')
                for inf in informs_items_list:
                    inf = inf.split(': ')
                    if inf[0] not in all_slots:
                        informs_correct = False
                        break
                    response['inform_slots'][inf[0]] = inf[1]

            requests_correct = True
            if len(chunks[2]) > 0:
                requests_key_list = chunks[2].split(', ')
                for req in requests_key_list:
                    if req not in all_slots:
                        requests_correct = False
                        break
                    response['request_slots'][req] = 'UNK'

            if intent_correct and informs_correct and requests_correct:
                break

        return response

    def _return_success(self):
        """
        Ask the user in console to input success (-1, 0 or 1) for (loss, neither loss nor win, win).

        Returns:
            int: Success: -1, 0 or 1
        """

        success = -2
        while success not in (-1, 0, 1):
            success = int(input('Success?: '))
        return success

    def step(self, agent_action):
        """
        Return the user's response, reward, done and success.

        Parameters:
            agent_action (dict): The current action of the agent

        Returns:
            dict: User response
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        # Assertions ----
        # No unk in agent action informs
        for value in agent_action['inform_slots'].values():
            assert value != 'UNK'
            assert value != 'PLACEHOLDER'
        # No PLACEHOLDER in agent_action at all
        for value in agent_action['request_slots'].values():
            assert value != 'PLACEHOLDER'
        # ---------------

        print('Agent Action: {}'.format(agent_action))

        done = False
        user_response = {'intent': '', 'request_slots': {}, 'inform_slots': {}}

        # First check round num, if equal to max then fail
        if agent_action['round'] == self.max_round:
            success = FAIL
            user_response['intent'] = 'done'
        else:
            user_response = self._return_response()
            success = self._return_success()

        if success == FAIL or success == SUCCESS:
            done = True

        assert 'UNK' not in user_response['inform_slots'].values()
        assert 'PLACEHOLDER' not in user_response['request_slots'].values()

        reward = reward_function(success, self.max_round)

        return user_response, reward, done, True if success is 1 else False
