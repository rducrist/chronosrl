"""ACCRL: action-chunked CRL, i.e. CRL predicting chunks of 3 actions."""

from ml_collections import config_dict

from chronosrl.algorithms.crl import CRL
from chronosrl.algorithms.crl import get_config as get_crl_config


def get_config(algorithm_name: str) -> config_dict.ConfigDict:
    """CRL hyperparameters with 3-step action chunks, training on every minibatch."""
    config = get_crl_config(algorithm_name)

    config.action_chunk_length = 3
    config.nr_sgd_batches = -1  # -1: use all minibatches instead of a random subset

    return config


# Same learner as CRL, only the config differs
ACCRL = CRL
