from .. import identify
from .. import trees


def _model_number(name):
    return int(name.split("-")[-1])


def summary(**kwargs):
    tls_version = kwargs["tls_version"]
    tree = trees.trees["hdt"][tls_version]

    # Sort the model names by number, instead of as a string (so model-10 comes
    # after model-2).
    model_names = sorted(tree.model_mapping.keys(), key=_model_number)

    summary = []
    for model in model_names:
        implementations = tree.model_mapping[model]

        result = {"Model": model}
        for weight_name, weight_function in identify.MODEL_WEIGHTS.items():
            result[weight_name] = weight_function(implementations)

        summary.append(result)
    return summary
