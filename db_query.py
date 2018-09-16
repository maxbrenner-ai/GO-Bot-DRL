class DBQuery:
    def __init__(self, database):
        self.database = database

    def fill_inform_slots(self, inform_slots_to_fill, current_inform_slots):
        return inform_slots_to_fill