"""module for running FL example on EMNIST data"""

import itertools
import logging
import fedjax
import jax
from src.run_fed_alg import run_federated_algorithm
from src.utils import load_config

FILENAME = "emnist"

logging.basicConfig(
    filename='logs/{}.log'.format(FILENAME),
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.DEBUG
)

# load config
model_params = load_config(FILENAME)

# load the EMNIST federated dataset
train, test = fedjax.datasets.emnist.load_data()

# creating a fedjax.Model object and initializing its parameters
model = fedjax.models.emnist.create_logistic_model()
# model = fedjax.models.emnist.create_conv_model()
rng = jax.random.PRNGKey(0)
init_params = model.init(rng) # weights and biases

# Creating a federated algorithm object and initializing its server state
grad_fn = fedjax.model_grad(model)
client_optimizer = fedjax.optimizers.adagrad(
    model_params['client']['learning_rate']
)
server_optimizer = fedjax.optimizers.adagrad(
    model_params['server']['learning_rate']
)
batch_hparams = fedjax.ShuffleRepeatBatchHParams(
    **model_params['client']['batch_hparams']
)
fed_alg = fedjax.algorithms.fed_avg.federated_averaging(
    grad_fn,
    client_optimizer,
    server_optimizer,
    batch_hparams
)
init_server_state = fed_alg.init(init_params)

# run fed alg
final_server_state, _ = run_federated_algorithm(
    federated_dataset=train,
    fed_alg=fed_alg,
    server_state=init_server_state,
    num_clients_per_round=[model_params['server']['num_clients_per_round']]\
            * model_params['server']['num_rounds'],
    rng=rng
)

# evaluation
params = final_server_state.params

# We select first 16 batches using itertools.islice
batched_test_data = list(itertools.islice(
    fedjax.padded_batch_federated_data(test, batch_size=128), 16))
batched_train_data = list(itertools.islice(
    fedjax.padded_batch_federated_data(train, batch_size=128), 16))

print('eval_test', fedjax.evaluate_model(model, params, batched_test_data))
print('eval_train', fedjax.evaluate_model(model, params, batched_train_data))
