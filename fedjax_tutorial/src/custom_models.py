"""A module for creating custom FedJAX Models for cifar10 example"""

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from fedjax.core import metrics
from fedjax.core import models


class Dropout(hk.Module):
    """Dropout haiku module."""

    def __init__(self, rate: float = 0.5):
        """Initializes dropout module.
        Args:
          rate: Probability that each element of x is discarded. Must be in [0, 1).
        """
        super().__init__()
        self._rate = rate

    def __call__(self, x: jnp.ndarray, is_train: bool):
        if is_train:
            return hk.dropout(rng=hk.next_rng_key(), rate=self._rate, x=x)
        return x


class ConvBatchNormDropoutModule(hk.Module):
    """
    Custom haiku module for CNN with dropout and batch normalization.
    Adapted from 'PyTorch: a 60 Minute Blitz'
    """

    def __init__(self, num_classes):
        super().__init__()
        self._num_classes = num_classes

    def __call__(self, x: jnp.ndarray, is_train: bool):
        x = hk.Conv2D(output_channels=6, kernel_shape=(5, 5), padding='VALID')(x)
        x = jax.nn.relu(x)
        #x = hk.BatchNorm(
        #        create_scale=False,
        #        create_offset=False
        #        decay_rate=0.99
        #    )(x, is_train)
        x = hk.MaxPool(
            window_shape=2,
            strides=2,
            padding='VALID'
        )(x)
        x = hk.Conv2D(output_channels=16, kernel_shape=(5, 5), padding='VALID')(x)
        x = jax.nn.relu(x)
        #x = hk.BatchNorm(
        #        create_scale=False,
        #        create_offset=False,
        #        decay_rate=0.99
        #    )(x, is_train)
        x = Dropout(rate=0.15)(x, is_train)
        x = hk.Flatten()(x)
        x = hk.Linear(120)(x)
        x = jax.nn.relu(x)
        #x = hk.BatchNorm(
        #        create_scale=False,
        #        create_offset=False,
        #        decay_rate=0.99
        #    )(x, is_train)
        x = Dropout(rate=0.25)(x, is_train)
        x = hk.Linear(84)(x)
        x = jax.nn.relu(x)
        #x = hk.BatchNorm(
        #        create_scale=False,
        #        create_offset=False,
        #        decay_rate=0.99
        #    )(x, is_train)
        x = Dropout(rate=0.5)(x, is_train)
        x = hk.Linear(self._num_classes)(x)
        return x


# Defines the expected structure of input batches to the model. This is used to
# determine the model parameter shapes.
_HAIKU_SAMPLE_BATCH = {
    'x': np.zeros((1, 32, 32, 3), dtype=np.float32),
    'y': np.zeros(1, dtype=np.float32)
}

_TRAIN_LOSS = lambda b, p: metrics.unreduced_cross_entropy_loss(b['y'], p)
_EVAL_METRICS = {
    'loss': metrics.CrossEntropyLoss(),
    'accuracy': metrics.Accuracy()
}

def create_cifar_conv_model() -> models.Model:
    """Creates CIFAR CNN model with dropout and batchnorm with haiku.
    Returns:
    Model
    """
    num_classes = 10

    def forward_pass(batch, is_train=True):
        return ConvBatchNormDropoutModule(num_classes)(batch['x'], is_train)

    transformed_forward_pass = hk.transform(forward_pass)
    return models.create_model_from_haiku(
        transformed_forward_pass=transformed_forward_pass,
        sample_batch=_HAIKU_SAMPLE_BATCH,
        train_loss=_TRAIN_LOSS,
        eval_metrics=_EVAL_METRICS,
        # is_train determines whether to apply dropout or not.
        train_kwargs={'is_train': True},
        eval_kwargs={'is_train': False})
