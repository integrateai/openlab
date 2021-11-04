"""A module for creating in-memory, custom FedJAX datasets"""

import tensorflow_datasets as tfds
import numpy as np
import fedjax

def split_data(n_clients, features, labels):
    """Randomly split dataset into n_clients and load them to a dict"""
    indices = np.random.randint(n_clients, size=features.shape[0])
    client_id_to_dataset_mapping = {}
    for i in range(n_clients):
        client_id_to_dataset_mapping[i] = {
            'x': features[indices == i, :, :, :],
            'y': labels[indices == i]
        }
    return client_id_to_dataset_mapping

def create_cifar_fedjax_dataset(n_clients=100):
    """Returns a CIFAR10 FedJAX.Dataset"""
    ds_train, ds_test = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=False
    )

    features, labels = list(ds_train.batch(50000).as_numpy_iterator())[0]
    features = features.astype(np.float32) / 255
    train_client_id_to_dataset_mapping = split_data(
        n_clients,
        features,
        labels
    )

    features, labels = list(ds_test.batch(10000).as_numpy_iterator())[0]
    features = features.astype(np.float32) / 255
    test_client_id_to_dataset_mapping = split_data(
        n_clients,
        features,
        labels
    )
    return (
        fedjax.InMemoryFederatedData(train_client_id_to_dataset_mapping),
        fedjax.InMemoryFederatedData(test_client_id_to_dataset_mapping)
    )
