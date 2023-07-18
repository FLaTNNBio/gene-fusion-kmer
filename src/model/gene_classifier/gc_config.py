from typing import Dict

from abc import ABCMeta
from abc import abstractmethod

from ..config import MyModelConfig


class GeneClassifierConfig(MyModelConfig, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
            self,
            model_dir: str,
            len_read: int = 150,
            len_kmer: int = 6,
            hyper_parameters: Dict[str, any] = None,
            n_classes: int = 1,
            **kwargs
    ):
        super().__init__(
            model_dir=model_dir,
            hyper_parameters={
                'len_read': len_read,
                'len_kmer': len_kmer,
                **hyper_parameters
            },
            n_classes=n_classes,
            **kwargs
        )

    @property
    def len_read(self) -> int:
        return self.hyper_parameters['len_read']

    @property
    def len_kmer(self) -> int:
        return self.hyper_parameters['len_kmer']
