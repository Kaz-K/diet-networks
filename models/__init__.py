from .DietNetworks import MLP
from .DietNetworks import DietNetworks
from .DietNetworks import ModifiedDietNetworks


def get_model(model_config):
    if model_config.model_name == 'MLP':
        model = MLP(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            output_size=model_config.output_size,
            use_dropout=model_config.use_dropout,
            use_reconstruction=model_config.use_reconstruction,
        )

    elif model_config.model_name == 'ModifiedMLP':
        model = MLP(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            output_size=model_config.output_size,
            use_dropout=model_config.use_dropout,
            use_reconstruction=model_config.use_reconstruction,
        )

    elif model_config.model_name == 'DietNetworks':
        model = DietNetworks(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            output_size=model_config.output_size,
            sample_size=model_config.sample_size,
            embedding_size=model_config.embedding_size,
            use_dropout=model_config.use_dropout,
            use_reconstruction=model_config.use_reconstruction,
        )

    elif model_config.model_name == 'ModifiedDietNetworks':
        model = ModifiedDietNetworks(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            output_size=model_config.output_size,
            sample_size=model_config.sample_size,
            embedding_size=model_config.embedding_size,
            use_dropout=model_config.use_dropout,
            use_reconstruction=model_config.use_reconstruction,
        )

    else:
        raise NotImplementedError

    print(model)
    return model
