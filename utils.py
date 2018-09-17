# Convert list to dict where the keys are the list elements, and the values are the indices of the elements in the list
def convert_list_to_dict(lst):
    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')
    return {k: v for v, k in enumerate(lst)}
