import json

def load_config(file_name):
    with open('configs/{}.json'.format(file_name)) as f:
        try:
            model_params = json.load(f)
        except:
            print("Cannot load content of {}.json".format(file_name))
            sys.exit(1)
    return model_params
