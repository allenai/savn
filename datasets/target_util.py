from datasets.constants import FULL_OBJECT_CLASS_LIST


def get_object_list(object_list_str):
    return FULL_OBJECT_CLASS_LIST


def get_object_index(target_objects, all_objects):
    idxs = []
    for o in target_objects:
        idxs.append(all_objects.index(o))
    return idxs
