#!/usr/bin/env python
# coding: utf-8


#Store the torch lighning data modules and classes 

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule ,Callback
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger
import torchvision.transforms as T
import numpy as np
import glob
import random
import wandb
from torch.utils.data import Dataset, DataLoader
import sys

# In[ ]:


# from __future__ import print_function, division
import os
# import pandas as pd
# from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.optim as optim
import glob

from torchvision import  utils
import torchvision.transforms as T
import torchvision
import torchvision.models as models


from PIL import Image
import random


# imports for examples
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(17)

from torchmetrics.classification import Accuracy, Recall
from torchmetrics import Precision, JaccardIndex
from torch.nn import  NLLLoss
# from sklearn.metrics import accuracy_score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# In[ ]:


class_to_idx = {'damaged_N': 0, 'damaged_Y': 1}
class_names = ['Not Damaged', 'Damaged']


# In[ ]:

class NOADataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]

        image = Image.open(image_filepath)
        label = image_filepath.split('/')[-2]
        
        if label == 'damaged_N':
            label = class_to_idx[label]
        elif label == 'damaged_Y': 
            label = class_to_idx[label]
        else:
            print('Label is incorrect')
            sys.exit(1)
        if self.transform is not None:
            image = self.transform(image) 
            
        return image, label

def custom_validation(condition_to_pass, errormessage):
    variable = False
    while not variable:
        if condition_to_pass is True:
            variable = True 
        else:
            print(errormessage)
            sys.exit(1)


class NOADataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './', H =180, W =180, data_mean = (0.4, 0.4 , 0.39) , data_sd = (0.22, 0.22, 0.22 ), num_workers = 0, test_only = None  ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([ T.Resize( size = (H, W) ), T.ToTensor(), T.Normalize(data_mean, data_sd) ])

        self.shape = (3, 180, 180)
        self.num_classes = 2
        self.glob_suffix = '/*.tif'
        self.test_only = test_only
  
    def setup(self, stage=None):
        train_data_path = self.data_dir + '/train'
        test_data_path = self.data_dir + '/test'

        train_image_paths_full = [] #to store image paths in list
        classes = [] #to store class values

        for data_path in glob.glob(train_data_path + '/*'): #search over both Y and N folders 
            # classes.append(data_path.split('/')[-1]) 
            # print(data_path)
            train_image_paths_full.append(glob.glob( data_path + self.glob_suffix) )  #only append tifs 

        custom_validation( len(train_image_paths_full) > 0 , errormessage= 'No training data available, batch size may be too large. ') 

        train_image_paths_full = np.concatenate(train_image_paths_full)
        train_image_paths_full = [x for x in train_image_paths_full]
        # print(train_image_paths)

        random.Random(4).shuffle(train_image_paths_full)

        #2.
        # split train valid from train paths (80,20)
        self.train_image_paths, self.valid_image_paths = train_image_paths_full[:int(0.8*len(train_image_paths_full))], train_image_paths_full[int(0.8*len(train_image_paths_full)):] 
        
        # print(self.valid_image_paths)

        #3.
        # create the test_image_paths
        test_image_paths = []
        for data_path in glob.glob(test_data_path + '/*'):
            test_image_paths.append(glob.glob(data_path + self.glob_suffix))
        # print(test_image_paths)

        self.test_image_paths = list(np.concatenate(test_image_paths).flat)

        custom_validation(( len(self.test_image_paths) > 0 ), errormessage= 'No test data available - batch size may be too large') 
        if self.test_only is None:
            custom_validation( ( len(self.train_image_paths) > len(self.test_image_paths) ), errormessage= 'More test data than train data: Test: {}, Train {}'.format(len(self.test_image_paths), len(self.train_image_paths)  )) 

        self.train_dataset = NOADataset(self.train_image_paths, self.transform)
        self.valid_dataset = NOADataset(self.valid_image_paths,self.transform) #test transforms are applied
        self.test_dataset = NOADataset(self.test_image_paths,self.transform)
        
        self.train_ids= [ x.split('/')[-1][:-4] for x in  self.train_image_paths]
        self.test_ids= [x.split('/')[-1][:-4]  for x in  self.test_image_paths]
        self.val_ids= [ x.split('/')[-1][:-4] for x in  self.valid_image_paths]

        print("Train size: {}\nValid size: {}\nTest size: {}".format(len(self.train_image_paths), len(self.valid_image_paths), len(self.test_image_paths)))
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers, pin_memory=True )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size )
    
    def test_shuffle_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = True  )
        




class Five_Conv_Model(pl.LightningModule):
    """5 layer Conv network, default to equal weights, and NNLoss
    """ 
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, optimiser = 'ADAM', weights = [ 1  , 1] , av_type = 'macro' , loss= NLLLoss):
        super().__init__()
    
        self.class_weights = torch.FloatTensor(weights)
        self.thresh  =  0.5
        self.optimiser = optimiser
        self.optimiser_dict = {'SGD': torch.optim.SGD , 'Adam': torch.optim.Adam, 'RMSProp': torch.optim.RMSprop, 'Adagrad' : torch.optim.Adagrad}
        # print(self.optimiser_dict[self.optimiser])
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.pool3 = torch.nn.MaxPool2d(2)
        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        #add metrics for tracking 
        self.accuracy = Accuracy()
        self.loss= loss(weight = self.class_weights)
        self.recall = Recall(num_classes=2, threshold=self.thresh, average = av_type)
        self.prec = Precision( num_classes=2, average = av_type)
        self.jacq_ind = JaccardIndex(num_classes=2)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.pool3(F.relu(self.conv5(x)))
        return x
    
    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, logger=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, logger=True)
        self.log('train_jacc', jac, on_step=True, on_epoch=True, logger=True)
        return loss
  
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_jacc', jac, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_jacc', jac, prog_bar=True)

        # self.log("conf_mat" ,  wandb.plot.confusion_matrix(probs=None, y_true=y, preds=preds, class_names=class_names) )
        return loss
    
    def configure_optimizers(self,):
        print('Optimise with {}'.format(self.optimiser) )
        optimizer = self.optimiser_dict[self.optimiser](self.parameters(), lr=self.learning_rate)
        
        return optimizer


    

class Five_Conv_ModelwithBatchNorm(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, optimiser = 'ADAM', weights = [1 ,1 ], av_type= 'micro', loss = NLLLoss ):
        super().__init__()
    
        self.class_weights = torch.FloatTensor(weights)
        self.thresh  =  0.5
        self.optimiser = optimiser
        self.optimiser_dict = {'SGD': torch.optim.SGD , 'Adam': torch.optim.Adam, 'RMSProp': torch.optim.RMSprop, 'Adagrad' : torch.optim.Adagrad}
        # print(self.optimiser_dict[self.optimiser])
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1)

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm34 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(256)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.pool3 = torch.nn.MaxPool2d(2)
        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        #add metrics for tracking 
        self.accuracy = Accuracy()
        self.loss= loss(weight = self.class_weights)
        self.recall = Recall(num_classes=2, threshold=self.thresh, average = av_type)
        self.prec = Precision( num_classes=2, average = av_type)
        self.jacq_ind = JaccardIndex(num_classes=2)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.batchnorm1 ( self.conv1(x)) )
        x = self.pool1(F.relu(self.batchnorm2 ( self.conv2(x))) ) 
        x = F.relu(self.batchnorm34 ( self.conv3(x)) )
        x = self.pool2(F.relu(self.batchnorm34 ( self.conv4(x))) ) 
        x = self.pool3(F.relu(self.batchnorm5 ( self.conv5(x))) ) 
        return x
    
    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, logger=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, logger=True)
        self.log('train_jacc', jac, on_step=True, on_epoch=True, logger=True)
        return loss
  
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_jacc', jac, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_jacc', jac, prog_bar=True)

        # self.log("conf_mat" ,  wandb.plot.confusion_matrix(probs=None, y_true=y, preds=preds, class_names=class_names) )
        return loss
    
    def configure_optimizers(self,):
        print('Optimise with {}'.format(self.optimiser) )
        optimizer = self.optimiser_dict[self.optimiser](self.parameters(), lr=self.learning_rate)
        
        return optimizer
        



class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                            for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                  preds[:self.num_samples], 
                                                  val_labels[:self.num_samples])]
            })

        


class ImagePredictionLogger_table(Callback):
    def __init__(self, samples , type):
        super().__init__()
        self.val_imgs, self.val_labels = samples
        self.type = type 
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # log table 
        my_table = wandb.Table(columns=[ "image", "label","prediction"] )
        for x, pred, y in zip(val_imgs, preds,  val_labels):
              my_table.add_data(wandb.Image(x) , y, pred )

        trainer.logger.experiment.log({self.type + '_pred_table' : my_table} )


class TransferLearning(pl.LightningModule):
    "Works for Resnet at the moment"
    def __init__(self, model, learning_rate, optimiser = 'Adam', weights = [ 1/2288  , 1/1500], av_type = 'macro' ):
        super().__init__()
        self.class_weights = torch.FloatTensor(weights)
        self.optimiser = optimiser
        self.thresh  =  0.5
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        #add metrics for tracking 
        self.accuracy = Accuracy()
        self.loss= nn.CrossEntropyLoss()
        self.recall = Recall(num_classes=2, threshold=self.thresh, average = av_type)
        self.prec = Precision( num_classes=2, average = av_type )
        self.jacq_ind = JaccardIndex(num_classes=2)
        

        # init model
        backbone = model
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify damage 2 classes
        num_target_classes = 2
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, logger=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, logger=True)
        self.log('train_jacc', jac, on_step=True, on_epoch=True, logger=True)
        return loss
  
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)


        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_jacc', jac, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_jacc', jac, prog_bar=True)

        return loss
    
    def configure_optimizers(self,):
        print('Optimise with {}'.format(self.optimiser) )
        # optimizer = self.optimiser_dict[self.optimiser](self.parameters(), lr=self.learning_rate)
                
                # Support Adam, SGD, RMSPRop and Adagrad as optimizers.
        if self.optimiser == "Adam":
            optimiser = optim.AdamW(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "SGD":
            optimiser = optim.SGD(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "Adagrad":
            optimiser = optim.Adagrad(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "RMSProp":
            optimiser = optim.RMSprop(self.parameters(), lr = self.learning_rate)
        else:
            assert False, f"Unknown optimizer: \"{self.optimiser}\""

        return optimiser




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False    
    

class TransferLearning_Ince(pl.LightningModule):
    "Inception V3 transfer leaning model "
    def __init__(self, learning_rate, optimiser = 'Adam', weights = [ 1  , 1], av_type = 'macro' ):
        super().__init__()
        self.class_weights = torch.FloatTensor(weights)
        self.optimiser = optimiser
        self.thresh  =  0.5
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        #add metrics for tracking 
        self.accuracy = Accuracy()
        self.loss= nn.CrossEntropyLoss()
        self.recall = Recall(num_classes=2, threshold=self.thresh, average = av_type)
        self.prec = Precision( num_classes=2, average = av_type )
        self.jacq_ind = JaccardIndex(num_classes=2)
        
        #Parameters for Inception V3
        feature_extract = True  #model finetunes and all features updated , if feature = True; only last layer will be updated 
        num_classes= 2 
        model_ft = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        #handle auxilliary net 
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        #handle primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        self.model = model_ft

    def forward(self, x):
        output  = self.model(x) 
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits , aux_outputs = self(x)
        loss1 = self.loss(logits, y)
        loss2 = self.loss(aux_outputs, y )
        loss = loss1 + 0.4*loss2
        
        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, logger=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, logger=True)
        self.log('train_jacc', jac, on_step=True, on_epoch=True, logger=True)
        return loss
  
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits  = self(x)
        loss = self.loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)


        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_jacc', jac, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        recall = self.recall(preds, y)
        precision = self.prec(preds, y)
        jac = self.jacq_ind(preds, y)


        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_jacc', jac, prog_bar=True)

        return loss
    
    def configure_optimizers(self,):
        print('Optimise with {}'.format(self.optimiser) )
        # optimizer = self.optimiser_dict[self.optimiser](self.parameters(), lr=self.learning_rate)
                
                # Support Adam, SGD, RMSPRop and Adagrad as optimizers.
        if self.optimiser == "Adam":
            optimiser = optim.Adam(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "SGD":
            optimiser = optim.SGD(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "Adagrad":
            optimiser = optim.Adagrad(self.parameters(), lr = self.learning_rate)
        elif self.optimiser == "RMSProp":
            optimiser = optim.RMSprop(self.parameters(), lr = self.learning_rate)
        else:
            assert False, f"Unknown optimizer: \"{self.optimiser}\""

        return optimiser
    
# def train_model(model_name, data_module, wand_proj_name ,  model_hparams, optimizer_name, optimizer_hparams, weights , av_type,max_epochs=50, save_name=None, test_model = False ):
#     """
#     Inputs:
#         model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
#         data_module - LightningModule that returns the train/ test/ val dataloaders 
#         wand_proj_name - name of the run for Wandb
#         data_dir - Directory of data 
#         save_name - Name to save the model with 
#         test_model - Set True if you want to run on test set

#     """

#     if save_name is None:
#         save_name = model_name

#     # Samples required by the custom ImagePredictionLogger callback to log image predictions.
#     val_samples = next(iter(data_module.val_dataloader()))
#     test_samples = next(iter(data_module.test_dataloader()))

#     # Initialize wandb logger
#     wandb_logger = WandbLogger(project=wand_proj_name, job_type='train' )


#     # Create a PyTorch Lightning trainer with the generation callback
#     trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                          # Where to save models
#                          max_epochs=max_epochs,
#                      progress_bar_refresh_rate=20, 
#                     gpus=1, 
#                     logger=wandb_logger,
#                     callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
#                                ImagePredictionLogger(val_samples), 
#                                ImagePredictionLogger_table( val_samples, 'val' ),
#                               #  ImagePredictionLogger_table(test_samples, 'test'),
#                               EarlyStopping(monitor="val_loss", mode="min")]

#                     )                                                   



#     pl.seed_everything(42) # To be reproducable
#     model = TransferLearning(model=model_name )

#     # model = ImagenetTransferLearning(learning_rate=0.001, optimiser= 'ADAM', av_type = 'macro' )
#     # Train the model ⚡🚅⚡
#     trainer.fit(model, data_module)

#     # Evaluate the model on the held-out test set ⚡⚡
#     if test_model is True:
#           trainer.test(dataloaders = data_module.test_dataloader())

#     # Close wandb run
#     wandb.finish()
  
##
## For param tuning

def param_sweep_train_model(config=None):
    """run with param tuning config  for resnet   
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        dm = NOADataModule(data_dir = '/content/gdrive/MyDrive/mres_proj/sandbox_data' , H= 224, W = 224, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
        dm.prepare_data()
        dm.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(dm.val_dataloader()))
        test_samples = next(iter(dm.test_dataloader()))

        wandb_logger = WandbLogger(project='Resnet152_param_sweep', job_type='train' )

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
        
        model = TransferLearning(model=models.resnet152(pretrained=True), learning_rate = config.learning_rate, optimiser = config.optimiser, weights = config.weights )

        trainer.fit(model, dm)

        # Evaluate the model on the held-out test set ⚡⚡

        trainer.test(dataloaders = dm.test_dataloader())

        # Close wandb run
        wandb.finish()

# def param_sweep_train_model_dense(config=None):
#     """run with param tuning config   for resnet   
#     """
#     with wandb.init(config=config):
#         # If called by wandb.agent, as below,
#         # this config will be set by Sweep Controller
#         config = wandb.config
        
#         dm = NOADataModule(data_dir = '/content/gdrive/MyDrive/mres_proj/sandbox_data' , H= 224, W = 224, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
#         dm.prepare_data()
#         dm.setup()

#         # Samples required by the custom ImagePredictionLogger callback to log image predictions.
#         val_samples = next(iter(dm.val_dataloader()))
#         test_samples = next(iter(dm.test_dataloader()))

#         wandb_logger = WandbLogger(project='desnenet_param_sweep', job_type='train' )

#         trainer = pl.Trainer(
#                         max_epochs=200,
#                         progress_bar_refresh_rate=20, 
#                         gpus=1, 
#                         logger=wandb_logger,
#                         callbacks=[#ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
#                                    ImagePredictionLogger(val_samples), 
#                                    ImagePredictionLogger_table( val_samples, 'val' ),
#                                   #  ImagePredictionLogger_table(test_samples, 'test'),
#                                   EarlyStopping(monitor="val_loss", mode="min")]

#                         )                                                   


#         pl.seed_everything(42) 
        
#         model = TransferLearning(model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True), learning_rate = config.learning_rate, optimiser = config.optimiser, weights = config.weights )

#         trainer.fit(model, dm)

#         # Evaluate the model on the held-out test set ⚡⚡

#         trainer.test(dataloaders = dm.test_dataloader())

#         # Close wandb run
#         wandb.finish()
        

def param_sweep_train_model_RS_FD(config=None):
    """run with param tuning config  for Resnet 50 with FUll data -- updated for resnet18   -- resnet152  
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        dm = NOADataModule(data_dir = '/content/gdrive/MyDrive/mres_proj/full_data' , H= 224, W = 224, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
        dm.prepare_data()
        dm.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(dm.val_dataloader()))
        test_samples = next(iter(dm.test_dataloader()))

        wandb_logger = WandbLogger(project='Resnet50_FD_param_sweep', job_type='train' )

        trainer = pl.Trainer(
                        max_epochs=50,
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
        
        model = TransferLearning(model=models.resnet152(pretrained=True), learning_rate = config.learning_rate, optimiser = config.optimiser, weights = config.weights , av_type = 'micro' )

        trainer.fit(model, dm)

        # Evaluate the model on the held-out test set ⚡⚡

        trainer.test(dataloaders = dm.test_dataloader())

        # Close wandb run
        wandb.finish()
        

def param_sweep_train_model_5conv(config=None):
    """run with param tuning config    
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        dm = NOADataModule(data_dir = '/content/gdrive/MyDrive/mres_proj/sandbox_data' , H= 180, W = 180, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
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

def param_sweep_train_model_5conv_BN(config=None):
    """run with param tuning config    
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        dm = NOADataModule(data_dir = '/content/gdrive/MyDrive/mres_proj/sandbox_data' , H= 180, W = 180, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
        dm.prepare_data()
        dm.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(dm.val_dataloader()))
        test_samples = next(iter(dm.test_dataloader()))

        wandb_logger = WandbLogger(project='5Conv_BN_paramsweep', job_type='train' )

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
        
        model = Five_Conv_ModelwithBatchNorm( dm.shape, dm.num_classes, learning_rate=config.learning_rate, optimiser = config.optimiser, weights = config.weights, av_type= 'macro' )

        trainer.fit(model, dm)

        # Evaluate the model on the held-out test set ⚡⚡

        trainer.test(dataloaders = dm.test_dataloader())

        # Close wandb run
        wandb.finish()


#

def param_sweep_train_model_I3_SD(config=None):
    """run with param tuning config for Inception V3  
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        dm = NOADataModule(data_dir = '/content/gdrive/MyDrive/mres_proj/sandbox_data' , H= 299, W = 299, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
        dm.prepare_data()
        dm.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(dm.val_dataloader()))
        test_samples = next(iter(dm.test_dataloader()))

        wandb_logger = WandbLogger(project='IV3_SB_param_sweep', job_type='train' )

        trainer = pl.Trainer(
                        max_epochs=50,
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
        
        model = TransferLearning_Ince(learning_rate = config.learning_rate, optimiser = config.optimiser, weights = config.weights , av_type = 'macro' )

        trainer.fit(model, dm)

        # Evaluate the model on the held-out test set ⚡⚡

        trainer.test(dataloaders = dm.test_dataloader())

        # Close wandb run
        wandb.finish()
        


def param_sweep_train_model_I3_FullData(config=None):
    """run with param tuning config for Inception V3  for full data set
    """
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        dm = NOADataModule(data_dir = '/content/gdrive/MyDrive/mres_proj/full_data' , H= 299, W = 299, data_mean = (0.416,0.416 , 0.36 ), data_sd = (0.215 ,0.21 , 0.21 ), batch_size=config.batch_size)
        
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
                                  #  ImagePredictionLogger_table(test_samples, 'test'),
                                  EarlyStopping(monitor="val_loss", mode="min")]

                        )                                                   


        pl.seed_everything(42) 
        
        model = TransferLearning_Ince(learning_rate = config.learning_rate, optimiser = config.optimiser, weights = config.weights , av_type = 'micro' )

        trainer.fit(model, dm)

        # Evaluate the model on the held-out test set ⚡⚡

        trainer.test(dataloaders = dm.test_dataloader())

        # Close wandb run
        wandb.finish()