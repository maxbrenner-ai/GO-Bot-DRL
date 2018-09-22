from collections import defaultdict


class DBQuery:
    def __init__(self, database):
        self.database = database
        # {frozenset: {string: int}} A dict of dicts
        self.cached_db_slot = defaultdict(dict)
        # {frozenset: {'#': {'slot': 'value'}}} A dict of dicts of dicts, a dict of DB sub-dicts
        self.cached_db = defaultdict(dict)

    def fill_inform_slots(self, inform_slots_to_fill, current_inform_slots):
        # Since this can be called even if nothign needs to be filled, return an empty dict if so
        if not inform_slots_to_fill:
            return {}

        # db_results is a dict of dict in the same exact format as the db, its just a subset of the db
        db_results = self.get_db_results(current_inform_slots)

        filled_informs = {}
        for slot in inform_slots_to_fill:
            values_dict = self._count_slot_values(slot, db_results)
            if values_dict:
                # Get key with max val (ie slot value with highest count of avail results)
                filled_informs[slot] = max(values_dict, key=values_dict.get)
            else:
                filled_informs[slot] = 'no match available'

        return inform_slots_to_fill

    def _count_slot_values(self, slot, db_subdict):
        slot_values = defaultdict(int)  # init to 0
        for id in db_subdict.keys():
            current_option_dict = db_subdict[id]
            # If there is a match
            if slot in current_option_dict.keys():
                slot_value = current_option_dict[slot]
                # This will add 1 to 0 if this is the first time this value has been encountered, or it will add 1
                # to whatever was already in there
                slot_values[slot_value] += 1

        return slot_values

    def get_db_results(self, constraints):
        constrain_keys = constraints.keys()
        # Todo: get rid of this filter in mine
        constrain_keys = filter(lambda k: k != 'numberofpeople', constrain_keys)
        # Cuz we dont want to constrain results for a slot that the usersim doesnt care about
        constrain_keys = [k for k in constrain_keys if constraints[k] != 'anything']

        inform_items = frozenset(constraints.items())
        cache_return = self.cached_db[inform_items]

        if cache_return is None:
            # If it is none then no matches fit with the constraints so return an empty dict
            return {}
        # if it isnt empty then return what it is
        if cache_return:
            return cache_return
        # else continue on

        available_options = {}
        for id in self.database.keys():
            current_option_dict = self.database[id]
            # First check if that databse movie actually contains the inform keys
            # Todo: again this assumes that if a slot key is not in a movie dict then it doesnt match
            if len(set(constrain_keys) - set(self.database[id].keys())) == 0:
                match = True
                # Now check all the slots values against the DB movie slots values and if there is a mismatch dont store
                for k, v in constraints.items():
                    # Todo: get rid of lower
                    if str(v).lower() != str(current_option_dict[k]).lower():
                        match = False
                if match:
                    # Update cache
                    self.cached_db[inform_items].update(current_option_dict)
                    available_options.update(current_option_dict)

        # if nothing avail then set it to none in cache
        if not available_options:
            self.cached_db[inform_items] = None

        return available_options

    def get_db_results_for_slots(self, current_informs):
        # The items (key, value) of the current informs are used as a key to the cached_db_slot
        inform_items = frozenset(current_informs.items())
        # A dict of the inform keys and their counts as stored (or not stored) in the cached_db_slot
        # Todo: Do time tests on multiple different DB to see if caching queries actually makes training faster overall
        cache_return = self.cached_db_slot[inform_items]
        # This is the same as checking if its not empty, ie the current informs have been used to query before
        if cache_return:
            return cache_return

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
