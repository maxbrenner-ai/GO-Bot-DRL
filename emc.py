import random
from dialogue_config import usersim_intents
import copy


class EMC:
    def __init__(self, db_dict, constants):
        # Dict of (string, list) where each key is the slot name and the list is of possible values
        self.movie_dict = db_dict
        self.slot_error_prob = constants['emc']['slot_error_prob']
        self.slot_error_mode = constants['emc']['slot_error_mode']  # 0 - 3
        self.intent_error_prob = constants['emc']['intent_error_prob']
        self.intents = usersim_intents

    def infuse_error(self, frame):
        informs_dict = frame['inform_slots']
        for key in list(frame['inform_slots'].keys()):
            if random.random() < self.slot_error_prob:
                if self.slot_error_mode == 0: # replace the slot_value only
                    self._slot_value_noise(key, informs_dict)
                elif self.slot_error_mode == 1: #replace slot and its values
                    self._slot_noise(key, informs_dict)
                elif self.slot_error_mode == 2:  # delete the slot
                    self._slot_remove(key, informs_dict)
                else: # Combine all three
                    rand_choice = random.random()
                    if rand_choice <= 0.33:
                        self._slot_value_noise(key, informs_dict)
                    elif rand_choice > 0.33 and rand_choice <= 0.66:
                        self._slot_noise(key, informs_dict)
                    else:
                        self._slot_remove(key, informs_dict)
        if random.random() < self.intent_error_prob:  # add noise for intent level
            frame['intent'] = random.choice(self.intents)
        return frame

    def _slot_value_noise(self, key, informs_dict):
        if key in self.movie_dict.keys():
            informs_dict[key] = random.choice(self.movie_dict[key])

    def _slot_noise(self, key, informs_dict):
        informs_dict.pop(key)
        random_slot = random.choice(list(self.movie_dict.keys()))
        informs_dict[random_slot] = random.choice(self.movie_dict[random_slot])

    def _slot_remove(self, key, informs_dict):
        informs_dict.pop(key)