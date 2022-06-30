#!/usr/bin/env python
# coding: utf-8

#Utils to store the Parameter sweep functions for Wandb Logging

#Imports 
import sys
import random
sys.path.append('~/HurricaneDamage/src')
from NOAAModules import NOADataModule, TransferLearning, Five_Conv_Model, Five_Conv_ModelwithBatchNorm, TransferLearning_Ince

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule ,Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import torchvision.models as models



#Update variables 
local_data_dir = '/content/gdrive/MyDrive/mres_proj/sandbox_data'
wand_proj_name = 'ResNet152_param_sweep'
resnet_model = models.resnet152(pretrained=True)




def param_sweep_train_model(config=None):
    """
    Parameter sweep in Wandb for ResNet model  
    """
    with wandb.init(config=config):
        config = wandb.config
        
        dm = NOADataModule(data_dir = local_data_dir , H= 224, W = 224, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
        dm.prepare_data()
        dm.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(dm.val_dataloader()))
        test_samples = next(iter(dm.test_dataloader()))

        wandb_logger = WandbLogger(project= wand_proj_name, job_type='train' )

        trainer = pl.Trainer(
                        max_epochs=200,
                        progress_bar_refresh_rate=20, 
                        gpus=1, 
                        logger=wandb_logger,
                        callbacks=[#ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                   ImagePredictionLogger(val_samples), 
                                   ImagePredictionLogger_table( val_samples, 'val' ),
                                  EarlyStopping(monitor="val_loss", mode="min")]

                        )                                                   


        pl.seed_everything(42) 
        
        model = TransferLearning(model= resnet_model , learning_rate = config.learning_rate, optimiser = config.optimiser, weights = config.weights )

        trainer.fit(model, dm)

        # Evaluate the model on the held-out test set
        trainer.test(dataloaders = dm.test_dataloader())

        # Close wandb run
        wandb.finish()



        

def param_sweep_train_model_5conv(config=None):
    """run with param tuning config    
    """
    with wandb.init(config=config):
        config = wandb.config
        
        dm = NOADataModule(data_dir = local_data_dir , H= 180, W = 180, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
        dm.prepare_data()
        dm.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(dm.val_dataloader()))
        test_samples = next(iter(dm.test_dataloader()))

        wandb_logger = WandbLogger(project='5Conv_paramsweep', job_type='train' )

        trainer = pl.Trainer(
                        max_epochs=200,
                        progress_bar_refresh_rate=20, 
                        gpus=1, 
                        logger=wandb_logger,
                        callbacks=[#ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                   ImagePredictionLogger(val_samples), 
                                   ImagePredictionLogger_table( val_samples, 'val' ),
                                  #  ImagePredictionLogger_table(test_samples, 'test'),
                                  EarlyStopping(monitor="val_loss", mode="min")]

                        )                                                   


        pl.seed_everything(42) 
        model = Five_Conv_Model( dm.shape, dm.num_classes, learning_rate=config.learning_rate, optimiser = config.optimiser, weights = config.weights, av_type= 'macro' )

        trainer.fit(model, dm)

        # Evaluate the model on the held-out test set ⚡⚡

        trainer.test(dataloaders = dm.test_dataloader())

        # Close wandb run
        wandb.finish()
  

##
##

def param_sweep_train_model_I3(config=None):
    """
    Run parameter tuning config for Inception V3 Architecture
    """
    with wandb.init(config=config):

        config = wandb.config
        
        dm = NOADataModule(data_dir = local_data_dir , H= 299, W = 299, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
        dm.prepare_data()
        dm.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(dm.val_dataloader()))
        test_samples = next(iter(dm.test_dataloader()))

        wandb_logger = WandbLogger(project='IV3_FB_param_sweep', job_type='train' )

        trainer = pl.Trainer(
                        max_epochs=50,
                        progress_bar_refresh_rate=20, 
                        gpus=1, 
                        logger=wandb_logger,
                        callbacks=[#ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                   ImagePredictionLogger(val_samples), 
                                   ImagePredictionLogger_table( val_samples, 'val' ),
                                  EarlyStopping(monitor="val_loss", mode="min")]

                        )                                                   


        pl.seed_everything(42) 
        
        model = TransferLearning_Ince(learning_rate = config.learning_rate, optimiser = config.optimiser, weights = config.weights , av_type = 'micro' )

        trainer.fit(model, dm)

        # Evaluate the model on the held-out test set
        trainer.test(dataloaders = dm.test_dataloader())

        # Close wandb run
        wandb.finish()