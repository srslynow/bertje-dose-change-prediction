from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Optional, Tuple, List

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.models.bert.modeling_bert import (
    BertForPreTraining
)

from dosechange_prediction.dataset import (
    DoseChangePredictionDataset,
    collate_fn,
)
from dosechange_prediction.serializers import PredictResponse, ArticleClassPrediction


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self, num_outputs: int = 1, hidden_size: int = 768, hidden_dropout_prob: float = 0.1
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_outputs)

    def forward(self, hidden_states, **kwargs):
        # hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class CaracalDoseChangePrediction(pl.LightningModule):
    def __init__(
        self,
        model_name,
        config,
        max_epochs: int = 8,
        learning_rate: float = 5e-6,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 100,
        weight_decay: float = 0.0,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        eval_splits: Optional[list] = None,
        accumulate_grad_batches: int = 1,
        **kwargs,
    ):
        super().__init__()
        # save model name to class var
        self.model_name = model_name
        # save config to class var and save hyperparams
        self.config = config
        self.save_hyperparameters()
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load the base model
        self.base_model = AutoModel.from_pretrained(model_name)
        # define the main classification head for the parent article_type prediction
        self.main_head = ClassificationHead(num_outputs=len(config.dose_change_id_to_label))
        # a list to keep validation outputs per epoch in
        self.validation_outputs = []

    def forward(self, input_ids, **kwargs):
        # set all tokens to local attention by default
        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=input_ids.device
        )
        # do a forward pass
        longformer_pred = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        # predict the main type
        main_head_pred = self.main_head(longformer_pred.pooler_output)
        return F.log_softmax(main_head_pred, dim=1)

    def predict(
        self, text: str
    ) -> Tuple[
        PredictResponse
    ]:
        with torch.no_grad():
            # tokenize input text
            input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            # do a forward pass
            logits_main = self.forward(input_ids)
            # output of model is log_softmax (for numeric stability)
            # do an exp to convert it so a probability-like distribution
            probabilities = torch.exp(logits_main)
            # get the predicted class idx
            main_class_id = torch.argmax(probabilities, dim=-1)[0].item()
            # get the probability of this class
            main_probabilities = probabilities[0].cpu().tolist()
            # a list where we'll store the predictions
            main_prediction = ArticleClassPrediction(
                id=main_class_id,
                name=self.config.dose_change_id_to_label[str(main_class_id)],
                selected=True,
                probability=main_probabilities[main_class_id],
            )
            all_predictions = []
            # loop over all main types
            for label_id, label_name in self.config.dose_change_id_to_label.items():
                all_predictions.append(
                    ArticleClassPrediction(
                        id=int(label_id),
                        name=label_name,
                        selected=label_id == str(main_class_id),
                        probability=main_probabilities[int(label_id)],
                    )
                )
            return PredictResponse(main_prediction=main_prediction, all_predictions=all_predictions)

    def training_step(self, batch, batch_idx):
        # unpack batch into x (input) and y (labels)
        x, y = batch
        # do a forward pass
        logits_main = self.forward(x)
        # calculate the main loss by comparing the predictions and the real labels
        loss = F.nll_loss(logits_main, y)
        # log the train loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # unpack batch into x (input) and y (labels)
        x, y = batch
        # do a forward pass
        logits_main = self.forward(x)
        # calculate the main loss by comparing the predictions and the real labels
        loss = F.nll_loss(logits_main, y)
        # do an argmax over the logits to get the predicted labels
        preds = torch.argmax(logits_main, dim=1)
        # get the accuracy comparing the predicted labels and the real labels
        acc = accuracy(preds, y, task='multiclass', num_classes=len(self.config.dose_change_id_to_label))
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        # save the val outputs
        # self.validation_outputs.append({'loss': loss, "preds": preds, "labels": y_main_pt})

    # def on_validation_epoch_end(self):
    #     if len(self.validation_outputs) > 0:
    #         # preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
    #         # labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
    #         loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
    #         self.log('val_loss', loss, prog_bar=True)
    #         self.log('val_acc', loss, prog_bar=True)
    #         self.validation_outputs.clear()

    def setup(self, stage):
        '''setup() function is only called when training/testing model, not in production'''
        if stage == 'fit':
            # load dependencies we only need while training
            from sqlalchemy.orm import load_only
            from webserver.main.db.deps import session_scope

            # start database session
            with session_scope() as db:
                # get the dataset
                df_examples = pd.read_csv('/workspace/data/dosechange_dataset.csv')
                # extract data as list of dicts
                examples_dicts = df_examples.to_dict('records')
                # only keep the text and the label
                examples = [(x['Dosisaanpassing'], self.config.dose_change_label_to_id[x['Antwoord']]) for x in examples_dicts]
                # split all articles into a training and validation set
                train_data, val_data = (examples[:int(0.8 * len(examples))], examples[int(0.8 * len(examples)):])
                # generate the Pytorch Datasets
                train_dataset = DoseChangePredictionDataset(train_data, tokenizer=self.tokenizer)
                val_dataset = DoseChangePredictionDataset(val_data, tokenizer=self.tokenizer)
                # create the examples batching function
                batch_collate_fn = partial(collate_fn, self.tokenizer)
                # create the samplers
                train_sampler = RandomSampler(train_dataset)
                val_sampler = SequentialSampler(val_dataset)
                # create dataloaders
                # unsupervised_dataloader = DataLoader(unsupervised_dataset, sampler=unsupervised_sampler, batch_size=train_batch_size, collate_fn=batch_collate_fn)
                self._train_dataloader = DataLoader(
                    train_dataset,
                    sampler=train_sampler,
                    batch_size=self.hparams.train_batch_size,
                    collate_fn=batch_collate_fn
                )
                self._validation_dataloader = DataLoader(
                    val_dataset,
                    sampler=val_sampler,
                    batch_size=self.hparams.eval_batch_size,
                    collate_fn=batch_collate_fn,
                )
                # Calculate total steps
                self.total_steps = (
                    (len(train_dataset) // self.hparams.train_batch_size)
                    // self.hparams.accumulate_grad_batches
                    * float(self.hparams.max_epochs)
                )

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._validation_dataloader

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.base_model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]


if __name__ == '__main__':
    from webserver.main.db.deps import session_scope
    from webserver.main.models.article_type_models import ArticleType
    from webserver.main.serializers import article_serializer
    from transformers import LongformerTokenizerFast

    # query for article types, are saved as metadata with the model
    # !!!! when changing the article_types in the database the model needs a retraining run !!!!
    with session_scope() as db:
        # query database for article types
        article_types = (
            db.query(ArticleType)
            .filter(ArticleType.use_in_training == True, ArticleType.parent_article_type_id == None)
            .all()
        )
        # convert sqlalchemy models to plain python structure (list/dicts)
        article_types = [article_serializer.ArticleType.from_orm(a).dict() for a in article_types]
    # specify our base model
    model_name = 'allenai/longformer-base-4096'
    # load the configuration of our base model
    config = LongformerConfig.from_pretrained(model_name)
    # attach the article_types to the configuration
    config.article_types = article_types
    # instantiate the model
    model = TypePrediction(config)

    # # load a tokenizer
    # tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
    # input_ids = tokenizer.encode('This is a sentence from [MASK] training data', return_tensors='pt')
    # model.forward(input_ids)

    AVAIL_GPUS = min(1, torch.cuda.device_count())
    trainer = pl.Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=3,
        progress_bar_refresh_rate=20,
    )
    trainer.fit(model)
    now = datetime.now().strftime('%Y%m%d_%H%M')
    model.save_pretrained(f'./saved_model_{now}')

    # loading a model
    model_path = '/workspace/saved_model_20210630_1712'
    # specify our base model
    # model_name = 'allenai/longformer-base-4096'
    # load the configuration of our base model
    config = LongformerConfig.from_pretrained(model_path)
    # instantiate the model
    model = TypePrediction.from_pretrained(model_path, config=config)
