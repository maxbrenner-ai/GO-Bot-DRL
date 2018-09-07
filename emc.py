class EMC:
    def __init__(self, level, type, error_amount):
        self.level = level  # Intent (1) or slot (2)
        self.type = type  # Out of the (1,2,3) of intent and (1,2,3,4) if slot
        self.error_amount = error_amount  # error chance [0,1]

    def infuse_error(self, frame):
        return frame

