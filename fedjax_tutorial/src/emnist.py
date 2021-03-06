"""module for running FL example on EMNIST data"""

import logging
import fedjax
import jax
from src.fed_alg_helper import run_federated_alg, eval_federated_alg
from src.utils import load_config

FILENAME = "emnist"

logging.basicConfig(
    filename='logs/{}.log'.format(FILENAME),
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.DEBUG
)

def main():
    """Main function for executing emnist example"""
    # load config
    config = load_config(FILENAME)
    model_params = config['model_params']

    # load the EMNIST federated dataset
    train, test = fedjax.datasets.emnist.load_data()

    # creating a fedjax.Model object and initializing its parameters
    model = fedjax.models.emnist.create_conv_model()
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
    final_server_state, _ = run_federated_alg(
        federated_dataset=train,
        fed_alg=fed_alg,
        server_state=init_server_state,
        num_clients_per_round=[model_params['server']['num_clients_per_round']]\
                * model_params['server']['num_rounds'],
        rng=rng
    )

    # evaluation of server model
    eval_federated_alg(
        model, final_server_state.params,
        train, test,
        batch_params=config['eval_params']
    )

if __name__ == "__main__":
    main()
