from collections import defaultdict


class DBQuery:
    def __init__(self, database):
        self.database = database
        self.cached_db_slot = defaultdict(dict)

    def fill_inform_slots(self, inform_slots_to_fill, current_inform_slots):
        return inform_slots_to_fill

    def results_for_slots(self, current_informs):
        # The items (key, value) of the current informs are used as a key to the cached_db_slot
        inform_items = frozenset(current_informs.items())
        # A dict of the inform keys and their counts as stored (or not stored) in the cached_db_slot
        slot_count_dict = self.cached_db_slot[inform_items]
        # This is the same as checking if its not empty, ie the current informs have been used to query before
        if slot_count_dict:
            return slot_count_dict

        # If it made it down here then a new query was made and it must add it to cached_db_slot and return it
        # init all key values with 0
        db_results = {key: 0 for key in current_informs.keys()}
        db_results['matching_all_constraints'] = 0

        for id in self.database.keys():
            all_slots_match = True
            for slot in current_informs.keys():
                # Unlike TC-bot i make it so if the user doesnt care about the value then its count goes up for every item in the db
                # Todo: Actually for now i will do what they do so i can test and compare results
                if current_informs[slot] == 'anything':
                    # db_results[slot] += 1
                    continue
                # Todo: So they count a inform slot key not being in the current movie in the db as a failure
                # meaning that they dont count it towards all matching and they dont add 1 to the count for that
                # slot itself.... idk if this is okay
                if slot in self.database[id].keys():
                    # TOdo: make it so lower isnt nec.
                    if current_informs[slot].lower() == self.database[id][slot].lower():
                        db_results[slot] += 1
                    else:
                        all_slots_match = False
                else:
                    all_slots_match = False
            if all_slots_match: db_results['matching_all_constraints'] += 1

        # update cache (set the empty dict)
        self.cached_db_slot[inform_items].update(db_results)
        return db_results