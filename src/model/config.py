from __future__ import annotations

from typing import Dict
from typing import List

from abc import ABCMeta
from abc import abstractmethod

from tabulate import tabulate
import json
import os

from . import MODEL_NAME
from . import CONF_FORMAT_NAME


class MyModelConfig(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
            self,
            model_dir: str,
            hyper_parameters: Dict[str, any] = None,
            n_classes: int = 1,
            **kwargs
    ):
        super().__init__()
        self.__model_dir: str = model_dir
        self.__model_name: str = MODEL_NAME
        self.__config_file_name: str = CONF_FORMAT_NAME.format(model_name=self.__model_name)
        assert hyper_parameters is not None
        self.__hyper_parameters: Dict[str, any] = hyper_parameters
        assert n_classes > 1
        self.__n_classes: int = n_classes

    def save_config(self) -> None:
        with open(os.path.join(self.__model_dir, self.__config_file_name), 'w') as handle:
            json.dump(
                {
                    'hyper_parameters': self.__hyper_parameters,
                    'n_classes': self.__n_classes
                },
                handle,
                indent=4
            )

    @classmethod
    def load_config(cls, model_dir: str) -> MyModelConfig:
        __config_file_name = CONF_FORMAT_NAME.format(model_name=MODEL_NAME)
        with open(os.path.join(model_dir, __config_file_name), 'r') as handle:
            __config: Dict[str, any] = json.load(handle)
        return cls(
            model_dir=model_dir,
            **__config['hyper_parameters'],
            n_classes=__config['n_classes']
        )

    @property
    def model_dir(self) -> str:
        return self.__model_dir

    @property
    def model_name(self) -> str:
        return self.model_name

    @property
    def hyper_parameters(self) -> Dict[str, any]:
        return self.__hyper_parameters

    @property
    def n_classes(self) -> int:
        return self.n_classes

    def __str__(self) -> str:
        table: List[List[str, any]] = [[parameter, value] for parameter, value in self.__hyper_parameters.items()]
        table_str: str = tabulate(
            tabular_data=table,
            headers=['hyper parameter', 'value'],
            tablefmt='psql',
            numalign='left'
        )
        return f'\n{table_str}\n'
