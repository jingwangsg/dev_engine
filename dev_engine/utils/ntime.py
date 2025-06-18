NTIME_COUNTER = {}


def global_once(name):
    global NTIME_COUNTER
    if name not in NTIME_COUNTER:
        NTIME_COUNTER[name] = 1
        return True
    return False
