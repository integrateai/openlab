"""convenience module for running fl algorithm"""

import logging
import fedjax
import jax
from src.utils import timing

@timing
def run_federated_algorithm(
        federated_dataset,
        fed_alg,
        server_state,
        num_rounds,
        num_clients_per_round,
        rng
):
    """
    Convenience function for performing the loop for running a federated
    algorithm consisting of client and server updates.

    return: ServerState
    """
    for j in range(num_rounds):
        print("Round: {}".format(j + 1))
        # sample clients to be included in current round
        client_sampler = fedjax.client_samplers.UniformGetClientSampler(
            federated_dataset,
            num_clients=num_clients_per_round,
            seed=j+1
        )
        sampled_clients_with_data = client_sampler.sample()
        logging.info(
            'Round: %d, Clients: %d',
            j+1,
            len(sampled_clients_with_data)
        )
        # client update & aggregated updates via FederatedAlgorithm.apply
        client_inputs = []
        for client_id, client_data, client_rng in sampled_clients_with_data:
            rng, use_rng = jax.random.split(rng)
            client_inputs.append((client_id, client_data, use_rng))
        updated_server_state, client_diagnostics = fed_alg.apply(
            server_state,
            client_inputs
        )
        server_state = updated_server_state
    return server_state, client_diagnostics
