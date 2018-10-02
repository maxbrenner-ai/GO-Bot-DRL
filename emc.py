import random


class EMC:
    def __init__(self, db_dict, constants):
        # Dict of (string, list) where each key is the slot name and the list is of possible values
        self.movie_dict = db_dict
        self.slot_error_prob = constants['emc']['slot_error_prob']
        self.slot_error_mode = constants['emc']['slot_error_mode']  # 0 - 3
        self.intent_error_prob = constants['emc']['intent_error_prob']

    # Todo: check if i even need to use return in these methods...
    def infuse_error(self, frame):
        informs_dict = frame['inform_slots']
        for key in informs_dict.keys():
            if random.random() < self.slot_error_prob:
                if self.slot_error_mode == 0: # replace the slot_value only
                    informs_dict = self._slot_value_noise(key, informs_dict)
                elif self.slot_error_mode == 1: #replace slot and its values
                    informs_dict = self._slot_noise(key, informs_dict)
                elif self.slot_error_mode == 2:  # delete the slot
                    informs_dict = self._slot_remove(key, informs_dict)
                else: # Combine all three
                    rand_choice = random.random()
                    if rand_choice <= 0.33:
                        informs_dict = self._slot_value_noise(key, informs_dict)
                    elif rand_choice > 0.33 and rand_choice <= 0.66:
                        informs_dict = self._slot_noise(key, informs_dict)
                    else:
                        informs_dict = self._slot_remove(key, informs_dict)
        frame['inform_slots'] = informs_dict
        return frame

    def _slot_value_noise(self, key, informs_dict):
        if key in self.movie_dict.keys():
            informs_dict[key] = random.choice(self.movie_dict[key])
        return informs_dict

    def _slot_noise(self, key, informs_dict):
        informs_dict.pop(key)
        random_slot = random.choice(self.movie_dict.keys())
        informs_dict[key] = random.choice(self.movie_dict[random_slot])
        return informs_dict

    def _slot_remove(self, key, informs_dict):
        informs_dict.pop(key)
        return informs_dict