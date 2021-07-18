import yaml
import tensorflow as tf


def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception as e:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.

    Args:
        config (dict): Configuration loaded from a yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the model hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    config.update(kwargs)
    return create_hparams(config)


def create_hparams(flags):
    """Create the model hyperparameters.

    Args:
        flags (dict): Dictionary with the model requirements.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    return tf.contrib.training.HParams(
        # data
        data_format=flags.get("data_format", None),
        iterator_type=flags.get("iterator_type", None),
        support_quick_scoring=flags.get("support_quick_scoring", False),
        wordEmb_file=flags.get("wordEmb_file", None),
        wordDict_file=flags.get("wordDict_file", None),
        userDict_file=flags.get("userDict_file", None),
        vertDict_file=flags.get("vertDict_file", None),
        subvertDict_file=flags.get("subvertDict_file", None),
        total_click=flags.get("total_click", None),
        total_test=flags.get("total_test", None),
        total_train=flags.get("total_train", None),
        # models
        title_size=flags.get("title_size", None),
        body_size=flags.get("body_size", None),
        word_emb_dim=flags.get("word_emb_dim", None),
        article_emb_size=flags.get("article_emb_size", None),
        word_size=flags.get("word_size", None),
        user_num=flags.get("user_num", None),
        vert_num=flags.get("vert_num", None),
        subvert_num=flags.get("subvert_num", None),
        his_size=flags.get("his_size", None),
        dense_dim=flags.get("dense_dim", None),
        npratio=flags.get("npratio"),
        dropout=flags.get("dropout", 0.0),
        attention_hidden_dim=flags.get("attention_hidden_dim", 200),
        # metonr
        neighbor_size=flags.get("neighbor_size", 4),
        neighbor_kind=flags.get("neighbor_kind", 4),
        dense_dim_list=flags.get("dense_dim_list", [128]),
        # gerl
        user_id_dim=flags.get("user_id_dim", 16),
        # dan
        rnn_layer_dim=flags.get("rnn_layer_dim", 128),
        # nrms
        head_num=flags.get("head_num", 4),
        head_dim=flags.get("head_dim", 100),
        # naml
        cnn_activation=flags.get("cnn_activation", None),
        dense_activation=flags.get("dense_activation", None),
        filter_num=flags.get("filter_num", 200),
        window_size=flags.get("window_size", 3),
        vert_emb_dim=flags.get("vert_emb_dim", 100),
        subvert_emb_dim=flags.get("subvert_emb_dim", 100),
        # lstur
        gru_unit=flags.get("gru_unit", 400),
        type=flags.get("type", "ini"),
        # npa
        user_emb_dim=flags.get("user_emb_dim", 50),
        # train
        learning_rate=flags.get("learning_rate", 0.001),
        loss=flags.get("loss", None),
        optimizer=flags.get("optimizer", "adam"),
        epochs=flags.get("epochs", 10),
        batch_size=flags.get("batch_size", 1),
        # show info
        show_step=flags.get("show_step", 1),
        metrics=flags.get("metrics", None),
    )
