from dev_engine import logging as log
from dill import dump, load
import os
import os.path as osp

object_store = dict()
obj_tmp_dir = osp.expanduser("~/.tmp/object")
os.makedirs(obj_tmp_dir, exist_ok=True)


def set_object(name, obj):
    log.debug(f"Setting object {name}")
    global object_store
    object_store[name] = obj


def get_object(name):
    global object_store
    return object_store[name]


def del_object(name):
    global object_store
    del object_store[name]


def save_object(name, obj):
    path = osp.join(obj_tmp_dir, name)
    log.debug(f"Saving object {name} to {path}")
    with open(path, "wb") as f:
        dump(obj, f)


def load_object(name):
    path = osp.join(obj_tmp_dir, name)
    log.debug(f"Loading object {name} from {path}")

    with open(path, "rb") as f:
        return load(f)

def object_in_disk(name):
    path = osp.join(obj_tmp_dir, name)
    return osp.exists(path)

def object_in_store(name):
    return name in object_store
    
