import json
import logging
from functools import wraps
from time import time

# timing wrapper modified code from 
# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:{} took: {:2.4f} sec'.format(f.__name__, te-ts))
        logging.info('func:{} took: {:2.4f} sec'.format(f.__name__, te-ts))
        return result
    return wrap

def load_config(file_name):
    with open('configs/{}.json'.format(file_name)) as f:
        try:
            model_params = json.load(f)
        except:
            print("Cannot load content of {}.json".format(file_name))
            sys.exit(1)
    return model_params
