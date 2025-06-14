from dev_engine import logging as log
from dill import dump, load
import os
import os.path as osp
import inspect
from pprint import pprint
from functools import wraps
import dill
import uuid
import traceback
from loguru import logger
from datetime import datetime

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


def save_locals(local_file):
    import sys

    frame = sys._getframe().f_back
    save_object(local_file, {k: v for k, v in frame.f_locals.items()})


def load_locals(local_file):
    local_dict = load_object(local_file)
    commands = "\n".join([f"{k} = {v}" for k, v in local_dict.items()])
    import sys

    frame = sys._getframe().f_back
    exec(commands, frame.f_globals, frame.f_locals)


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


def capture_calls(dump_path="debug", fault_only=False):

    def decorator(fn):
        print(f"Capture calls for {fn.__name__}")

        @wraps(fn)
        def wrapper(*args, **kwargs):
            uid = datetime.now().strftime("%Y%m%d_%H%M%S")

            exception = None
            try:
                ret = fn(*args, **kwargs)
            except Exception as e:
                exception = e
            
            if fault_only and exception is None:
                return ret

            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)

            traceback.print_exc()

            logger.info(f"================ PARAMS ================")
            logger.info(params)

            _path = osp.join(dump_path, f"{fn.__name__}_{uid}.replay")
            with open(_path, "wb") as f:
                dill.dump({"fn": fn, "params": params}, f)

            logger.info(f"Dumped {fn.__name__} inputs  => {osp.abspath(_path)}")

            return ret

        return wrapper


    return decorator
