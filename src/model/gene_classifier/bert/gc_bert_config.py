from ..gc_config import GeneClassifierConfig


class GCBertModelConfig(GeneClassifierConfig):
    def __init__(
            self,
            model_dir: str,
            len_read: int = 150,
            len_kmer: int = 6,
            hidden_size: int = 768,
            n_hidden_layers: int = 12,
            n_attention_heads: int = 12,
            intermediate_size: int = 3072,
            hidden_act: str = 'gelu',
            hidden_dropout_prob: float = 0.1,
            attention_dropout_prob: float = 0.1,
            max_position_embeddings: int = 512,
            type_vocab_size: int = 2,
            initializer_range: int = 0.02,
            n_classes: int = 1,
            **kwargs
    ):
        # call super class
        super().__init__(
            model_dir=model_dir,
            len_read=len_read,
            len_kmer=len_kmer,
            hyper_parameters={
                'hidden_size': hidden_size,
                'n_hidden_layers': n_hidden_layers,
                'n_attention_heads': n_attention_heads,
                'intermediate_size': intermediate_size,
                'hidden_act': hidden_act,
                'hidden_dropout_prob': hidden_dropout_prob,
                'attention_dropout_prob': attention_dropout_prob,
                'max_position_embeddings': max_position_embeddings,
                'type_vocab_size': type_vocab_size,
                'initializer_range': initializer_range
            },
            n_classes=n_classes,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return self.hyper_parameters['vocab_size']

    @property
    def hidden_size(self) -> int:
        return self.hyper_parameters['hidden_size']

    @property
    def n_hidden_layers(self) -> int:
        return self.hyper_parameters['n_hidden_layers']

    @property
    def intermediate_size(self) -> int:
        return self.hyper_parameters['intermediate_size']

    @property
    def hidden_act(self) -> str:
        return self.hyper_parameters['hidden_act']

    @property
    def hidden_dropout_prob(self) -> float:
        return self.hyper_parameters['hidden_dropout_prob']

    @property
    def attention_dropout_prob(self) -> float:
        return self.hyper_parameters['attention_dropout_prob']

    @property
    def max_position_embeddings(self) -> int:
        return self.hyper_parameters['max_position_embeddings']

    @property
    def type_vocab_size(self) -> int:
        return self.hyper_parameters['type_vocab_size']

    @property
    def initializer_range(self) -> float:
        return self.hyper_parameters['initializer_range']
