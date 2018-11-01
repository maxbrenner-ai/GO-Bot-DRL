from dialogue_config import FAIL, NO_OUTCOME, SUCCESS, usersim_intents, all_slots
import re


class User:
    def __init__(self, constants):
        self.max_round = constants['run']['max_round_num']

    def reset(self):
        return self._return_response()

    def _return_response(self):
        # Format must be like this: request/moviename: MIB, date: friday/time, cost, reviews
        # or inform/moviename: zooptopia/
        # or request//time
        # or done//
        # intents, informs keys and values, and request keys and values cannot conain / , :
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

    def _return_succ(self):
        # Either -1, 0, 1
        succ = -2
        while succ not in (-1, 0, 1):
            succ = int(input('Succ.: '))
        return succ

    def step(self, agent_action, round_num):
        # Assetions
        # No unk in agent action informs
        for value in agent_action['inform_slots'].values():
            assert value != 'UNK'
            assert value != 'PLACEHOLDER'
        # No PLACHEOLDERin agent at all
        for value in agent_action['request_slots'].values():
            assert value != 'PLACEHOLDER'

        print('Agent Action: {}'.format(agent_action))

        done = False
        user_response = {'intent': '', 'request_slots': {}, 'inform_slots': {}}
        # First check round num, if past max then fail
        if round_num > self.max_round:
            succ = FAIL
            user_response['intent'] = 'done'
        else:
            user_response = self._return_response()
            succ = self._return_succ()

        if succ == FAIL or succ == SUCCESS:
            done = True

        assert 'UNK' not in user_response['inform_slots'].values()
        assert 'PLACEHOLDER' not in user_response['request_slots'].values()

        reward = self._reward_function(succ)

        return user_response, reward, done, True if succ is 1 else False

    def _reward_function(self, succ):
        reward = -1
        if succ == FAIL:
            reward += -self.max_round
        elif succ == SUCCESS:
            reward += 2 * self.max_round
        return reward
