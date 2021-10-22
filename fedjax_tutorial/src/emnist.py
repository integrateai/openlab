import fedjax
import jax
from run_fed_alg import run_federated_algorithm
from utils import load_config

FILENAME = "emnist"

# load config
model_params = load_config(FILENAME)

# load the EMNIST federated dataset
train, test = fedjax.datasets.emnist.load_data()

# example of a readily available fedjax.Model
model = fedjax.models.emnist.create_logistic_model()

# initial params for the fedjax.Model
rng = jax.random.PRNGKey(0)
init_params = model.init(rng) # weights and biases

# Creating a federated algorithm object
grad_fn = fedjax.model_grad(model) 
client_optimizer = fedjax.optimizers.sgd(
    model_params['client']['learning_rate']
)
server_optimizer = fedjax.optimizers.sgd(
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

run_federated_algorithm(
    federated_dataset=train,
    fed_alg=fed_alg,
    server_state=init_server_state,
    num_rounds=model_params['server']['num_rounds'],
    num_clients_per_round=model_params['server']['num_clients_per_round'],
    rng = rng
)
