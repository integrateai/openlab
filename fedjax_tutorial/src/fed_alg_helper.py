"""convenience module for running fl algorithm"""

import itertools
import logging
import fedjax
import jax
from src.utils import timing

@timing
def run_federated_alg(
        federated_dataset, fed_alg, server_state,
        num_clients_per_round, rng
):
    """
    Convenience function for performing the loop for running a federated
    algorithm consisting of client and server updates.

    return: ServerState
    """
    for j, num_clients in enumerate(num_clients_per_round):
        print("Round: {}".format(j + 1))
        # sample clients to be included in current round
        client_sampler = fedjax.client_samplers.UniformGetClientSampler(
            federated_dataset,
            num_clients=num_clients,
            seed=j+1
        )
        logging.info(
            'Round: %d, Clients: %d',
            j+1,
            num_clients
        )
        # client update & aggregated updates via FederatedAlgorithm.apply
        client_inputs = []
        for client_id, client_data, _ in client_sampler.sample():
            rng, use_rng = jax.random.split(rng)
            client_inputs.append((client_id, client_data, use_rng))
        updated_server_state, client_diagnostics = fed_alg.apply(
            server_state,
            client_inputs,
        )
        server_state = updated_server_state
    return server_state, client_diagnostics

@timing
def eval_federated_alg(model, params, train, test, batch_params):
    """
    Convenience function for computing performance metrics of a model
    on train and test federated datsets.
    """
    batched_test = list(
        itertools.islice(
            fedjax.padded_batch_federated_data(
                test,
                batch_size=batch_params['test']['size']
            ),
            batch_params['test']['num_batches']
        )
    )
    batched_train = list(
        itertools.islice(
            fedjax.padded_batch_federated_data(
                train,
                batch_size=batch_params['train']['size']
            ),
            batch_params['train']['num_batches']
        )
    )
    print('eval_test', fedjax.evaluate_model(model, params, batched_test))
    print('eval_train', fedjax.evaluate_model(model, params, batched_train))
    logging.info(
        "performed evaluation model on batched train and test dataset"
    )
