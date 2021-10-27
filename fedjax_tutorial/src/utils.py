"""utility functions commonly shared by example scripts"""

import json
import logging
import sys
from functools import wraps
from time import time

# timing wrapper modified code from
# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(func):
    """time wrapper for calculating execution time"""
    @wraps(func)
    def wrap(*args, **kw):
        time_start = time()
        result = func(*args, **kw)
        time_end = time()
        print('func:{} took: {:2.4f} sec'\
            .format(func.__name__, time_end-time_start))
        logging.info('func:{} took: {:2.4f} sec'\
            .format(func.__name__, time_end-time_start))
        return result
    return wrap

def load_config(file_name):
    """utility function to load json config"""
    with open('configs/{}.json'.format(file_name)) as f:
        try:
            model_params = json.load(f)
        except: # pylint: disable=bare-except
            print("Cannot load content of {}.json".format(file_name))
            sys.exit(1)
    return model_params
