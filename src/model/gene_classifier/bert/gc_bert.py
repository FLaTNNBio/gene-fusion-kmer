from typing import Optional
from typing import Tuple
from typing import Dict

import torch
import torch.nn as nn

from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert import BertModel

from src.model import MyModel
from src.model.gene_classifier import GCBertModelConfig


class GCBert(MyModel):
    def __init__(
            self,
            model_dir: str,
            model_name: str = 'model',
            config: GCBertModelConfig = None,
            n_classes: int = 1,
            weights: Optional[torch.Tensor] = None
    ):
        # call super class
        super().__init__(
            model_dir=model_dir,
            model_name=model_name,
            config=config,
            n_classes=n_classes,
            weights=weights
        )

        # init configuration of model
        __bert_config = BertModel(
            **config.hyper_parameters
        )

        # create model from configuration
        self.bert = BertModel(__bert_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.n_classes if self.n_classes > 2 else 1)

        # init loss function
        if self.n_classes == 2:
            self.__loss = BCEWithLogitsLoss(pos_weight=weights)
        else:
            self.__loss = CrossEntropyLoss(weight=weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        # call bert forward
        _, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # dropout and linear output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)

        return outputs

    def load_data(
            self,
            batch, device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # prepare input of batch for classifier
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        target = batch['label'].to(device)
        # return Dict
        return {
                   'input_ids': input_ids,
                   'attention_mask': attention_mask,
                   'token_type_ids': token_type_ids
               }, target

    def step(
            self,
            inputs: Dict[str, torch.Tensor]
    ) -> any:
        # call self.forward
        return self(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )

    def compute_loss(
            self,
            target: torch.Tensor,
            output: torch.Tensor
    ) -> torch.Tensor:
        if self.n_classes == 2:
            return self.__loss(output.view(-1), target.view(-1))
        else:
            return self.__loss(output.view(-1, self.n_classes), target.view(-1))
