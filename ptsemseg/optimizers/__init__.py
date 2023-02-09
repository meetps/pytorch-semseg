import logging

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

logger = logging.getLogger("ptsemseg")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(cfg):
    if cfg["training"]["optimizer"] is None:
        logger.info("Using SGD optimizer")
        return SGD

    else:
        opt_name = cfg["training"]["optimizer"]["name"]
        if opt_name not in key2opt:
            raise NotImplementedError(f"Optimizer {opt_name} not implemented")

        logger.info(f"Using {opt_name} optimizer")
        return key2opt[opt_name]
