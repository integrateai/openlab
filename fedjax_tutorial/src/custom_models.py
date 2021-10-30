"""A module for creating custom FedJAX Models for cifar10 example"""

from jax.experimental import stax
from jax import random
import jax.numpy as jnp
from fedjax.core import metrics
from fedjax.core import models


_STAX_SAMPLE_SHAPE = (-1, 32, 32, 3)
_TRAIN_LOSS = lambda b, p: metrics.unreduced_cross_entropy_loss(b['y'], p)
_EVAL_METRICS = {
    'loss': metrics.CrossEntropyLoss(),
    'accuracy': metrics.Accuracy()
}

def create_stax_cifar_conv_model() -> models.Model:
    """
    Create fedjax Model conv net from a stax libraries.
    Adapted CNN from 'PyTorch: A 60 Minute Blitz'
    """
    num_classes = 10
    stax_init, stax_apply = stax.serial(
        stax.Conv(
            out_chan=6,
            filter_shape=(5, 5),
            strides=(1, 1),
            padding='same'
        ),
        stax.Relu,
        stax.BatchNorm(),
        stax.MaxPool((2, 2)),
        stax.Conv(
            out_chan=16,
            filter_shape=(5, 5),
            strides=(1, 1),
            padding='same'
        ),
        stax.Relu,
        stax.BatchNorm(),
        stax.Flatten,
        ProperDropout(0.15),
        stax.Dense(120),
        stax.Relu,
        stax.BatchNorm(axis=(0, 1)),
        ProperDropout(0.25),
        stax.Dense(84),
        stax.Relu,
        stax.BatchNorm(axis=(0, 1)),
        ProperDropout(0.50),
        stax.Dense(num_classes),
    )
    return models.create_model_from_stax(
        stax_init=stax_init,
        stax_apply=stax_apply,
        sample_shape=_STAX_SAMPLE_SHAPE,
        train_loss=_TRAIN_LOSS,
        eval_metrics=_EVAL_METRICS,
        train_kwargs={'mode': 'train'},
        eval_kwargs={'mode': 'score'}
    )


def ProperDropout(rate): # pylint: disable=invalid-name
    """Adapted stax.Dropout to function when mode != 'train'"""
    def init_fun(rng, input_shape):
        return input_shape, ()
    def apply_fun(params, inputs, **kwargs):
        mode = kwargs.get('mode', 'train')
        rng = kwargs.get('rng', None)
        if (rng is None) & (mode == 'train'):
            msg = (
                "Dropout layer requires apply_fun to be called with a PRNG key "
                "argument. That is, instead of `apply_fun(params, inputs)`, call "
                "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                "jax.random.PRNGKey value."
            )
            raise ValueError(msg)
        if mode == 'train':
            keep = random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        return inputs
    return init_fun, apply_fun
