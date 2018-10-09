import logging
import os
from collections import defaultdict
from pathlib import Path

def learn_models(directory):
    """Learn the complete tree from all the different models.

    Args:
        directory: A directory that contains the output of StateLearner. It
            expects the directory to contain directories with the names of the
            different TLS implementations, where each TLS directory contains a
            file 'learnedModel.dot'. Will skip implementations where this file
            is absent.

    Returns:
        model_tree: A networkx tree containing the paths for all models.
    """
    logger = logging.getLogger()

    # Read and deduplicate the models
    server_dirs = [f.name for f in os.scandir(directory) if f.is_dir()]
    models = defaultdict(list)

    for server in server_dirs:
        try:
            model_root = Path(directory)
            with (model_root / server / 'learnedModel.dot').open() as f:
                models[f.read()].append(server)

                logger.info(f'Found model for: {server}')
        except OSError:
            logger.warning(f'Could not find model for: {server}')

    return models
