import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

max_prediction_length = 6
max_encoder_length = 24
batch_size = 128  # set this between 32 to 128

def read_temporal_ds(filename : str) :
    ds1 = pd.read_csv(filename, delimiter='\t')
    ds = ds1.fillna(0)
    
    ts = TimeSeriesDataSet(
    ds,
    time_idx="N",
    target="search_progress",
    group_ids=["filename"],
    min_encoder_length=max_encoder_length // 4,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["filename"],
    static_reals=["h0"],
    time_varying_known_categoricals=[],
    variable_groups={},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["N"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "g", "h", "f", "hmin", "N_of_hmin",
        "a0_g", "a0_h", "a0_f", "a0_hmin", "a0_N_of_hmin",
        "a1_g", "a1_h", "a1_f", "a1_hmin", "a1_N_of_hmin",
        "a2_g", "a2_h", "a2_f", "a2_hmin", "a2_N_of_hmin",
        "a0_h0", "a1_h0", "a2_h0"        
    ],
    target_normalizer=GroupNormalizer(
        groups=["filename"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    )
    return ts

tds_train = read_temporal_ds("train_ds.txt")
tds_val = read_temporal_ds("val_ds.txt")
test_val = read_temporal_ds("test_ds.txt")


train_dataloader = tds_train.to_dataloader(train=True, batch_size=batch_size, num_workers=16)
val_dataloader = tds_val.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=16)

pl.seed_everything(42)
trainer = pl.Trainer(
    accelerator="gpu",
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    tds_train,
    learning_rate=0.2,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

torch.set_float32_matmul_precision('medium')
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)