import os
import importlib
from .base import BaseTrainer

import argparse

TRAINER_REGISTRY = {'base':BaseTrainer}

__all__ = [
    "BaseTrainer"
]

def register_trainer(name):
    def register_trainer_cls(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate trainer ({})".format(name))
        if not issubclass(cls, BaseTrainer):
            raise ValueError(
                "Model ({}: {}) must extend Trainer".format(name, cls.__name__)
            )
        TRAINER_REGISTRY[name] = cls

        return cls
    return register_trainer_cls


trainer_dir = os.path.dirname(__file__)
for file in os.listdir(trainer_dir):
    path = os.path.join(trainer_dir, file)
    if (
        file != "hotflip.py"
        and not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("trainer." + model_name)

        if model_name in TRAINER_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group("Named Trainer")
            group_archs.add_argument(
                "--trainer", choices=TRAINER_REGISTRY[model_name]
            )
            group_args = parser.add_argument_group("Additional command-line arguments")
            TRAINER_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + "_parser"] = parser



