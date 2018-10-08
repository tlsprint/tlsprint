def learn(directory):
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
    subdirs = [f.name for f in os.scandir(directory) if f.is_dir()]
    return subdirs
