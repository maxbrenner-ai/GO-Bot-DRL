from collections import defaultdict


class DBQuery:
    def __init__(self, database):
        self.database = database
        # {frozenset: {string: int}} A dict of dicts
        self.cached_db_slot = defaultdict(dict)
        # {frozenset: {'#': {'slot': 'value'}}} A dict of dicts of dicts, a dict of DB sub-dicts
        self.cached_db = defaultdict(dict)

    def fill_inform_slots(self, inform_slots_to_fill, current_inform_slots):
        # db_results is a dict of dict in the same exact format as the db, its just a subset of the db
        db_results = self._get_db_results(current_inform_slots)
        return inform_slots_to_fill

    def _get_db_results(self, inform_slots):
        constrain_keys = inform_slots.keys()
        # Todo: get rid of this filter in mine
        constrain_keys = filter(lambda k: k != 'numberofpeople', constrain_keys)
        # Cuz we dont want to constrain results for a slot that the usersim doesnt care about
        constrain_keys = [k for k in constrain_keys if inform_slots[k] != 'anything']

        inform_items = frozenset(inform_slots.items())
        cache_return = self.cached_db[inform_items]

        if cache_return is None:
            # If it is none then no matches fit with the constraints so return an empty dict
            return dict()
        # if it isnt empty then return what it is
        if cache_return:
            return cache_return
        # else continue on

        available_movies = {}
        for id in self.database.keys():
            current_movie_dict = self.database[id]
            # First check if that databse movie actually contains the inform keys
            # Todo: again this assumes that if a slot key is not in a movie dict then it doesnt match
            if len(set(constrain_keys) - set(self.database[id].keys())) == 0:
                match = True
                # Now check all the slots values against the DB movie slots values and if there is a mismatch dont store
                for k, v in inform_slots.items():
                    # Todo: get rid of lower
                    if str(v).lower() != str(current_movie_dict[k]).lower():
                        match = False
                if match:
                    # Update cache
                    self.cached_db[inform_items].update(current_movie_dict)
                    available_movies.update(current_movie_dict)

        # if nothing avail then set it to none in cache
        if not available_movies:
            self.cached_db[inform_items] = None

        return available_movies

    def results_for_slots(self, current_informs):
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
