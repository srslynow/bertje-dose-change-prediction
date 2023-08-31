import os
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoConfig, AutoModel, AutoTokenizer

from webserver.main.db.deps import session_scope
from webserver.main.models.article_type_models import ArticleType
from webserver.main.serializers import article_serializer
from dosechange_prediction.config import dose_change_id_to_label, dose_change_label_to_id
from dosechange_prediction.model_pt_lightning import CaracalDoseChangePrediction


def train_model():
    # specify our base model
    model_name = 'GroNLP/bert-base-dutch-cased'
    # load the configuration of our base model
    config = AutoConfig.from_pretrained(model_name)
    # attach the article_types to the configuration
    config.model_name = model_name
    config.dose_change_id_to_label = dose_change_id_to_label
    config.dose_change_label_to_id = dose_change_label_to_id
    # instantiate the model
    model = CaracalDoseChangePrediction(model_name=model_name, config=config)
    # create the tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger('.')
    csv_logger = pl_loggers.CSVLogger('.', flush_logs_every_n_steps=20)
    # create the checkpointer
    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoints",
        filename='epoch={epoch}-validation_acc={val_acc:.2f}',
        save_top_k=2,
        monitor="val_acc",
        mode='max',
        verbose=True,
        save_weights_only=True
    )
    # create the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=10,
        callbacks=[checkpoint_callback],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=20
    )
    trainer.fit(model)

    # # save_path = '/data/caracal-dosechange/longformer/' + datetime.now().strftime(f'%Y%m%d_%H%M')
    # save_path = '/workspace/data/caracal-dosechange/longformer/' + datetime.now().strftime(f'%Y%m%d_%H%M')
    # # create the directory
    # # os.makedirs(save_path)
    # model.save_pretrained(save_path)
