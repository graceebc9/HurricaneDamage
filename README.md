<img width="100" alt="Cam logo" src= https://seeklogo.com/images/U/university-of-cambridge-logo-E6ED593FBF-seeklogo.com.png >          <img width="100" alt="RMS logo" src=https://www.burstorm.com/wp-content/uploads/RMS-logo-final.png>    

This is the repository for Grace Colverds AI4ER MRES project titled: 'A deep learning approach to post-hurrciane automatic damage classification'

## 1. Overview

This repository contains all code written for the AI4ER MRES Report.

This project focuses on classifying damage in the aftermath of Hurricanes. Problem is formualted as patch-based image classification on patches generated from imagery sourced by the National Oceanic and Atmospheric Administration (NOAA), alongside building footprints and damage labels provided by Risk Management Solutions (RMS). The two hurricanes looked at are Hurricane Ida (dataset used for model creation and evaluation) and Hurricane Delta (used to test model transferabiilty and performance after limited finetuning).

Patches are created per building footprint by first aligning building footprint with RBG imagery using lat/lons, then generating a square patch around the building, using an area that is scaled for the median house size out of all buildings received. This enables model to understand scale of building, although for some of the larger buildings surrounding area is cropped out. 

4 datatsets were created (all genreated to be balanced between damaged and undamaged patches- using number of damaged buildings as the limiting factor) : 
### Ida imagery: 
- 1a: A small 'sandbox' data set with which to quickly test model architectures 
- 2a: A 'full' dataset containing all of the damaged buidlings
### Delta imagery:
- 2a: a dataset containing all damaged buildings to test model on
- 2b: 2a but first taking out 300 patches to quickly finetune model on before testing 

We train a convolutional neural network to predict a label of 'damaged' or 'undamaged' per patch using dataset 1a and 1b. Predictions for each patch are then aggreagated to higher regional levels in order to create damage maps both at the block and census tract level. 

The best performing model is then evaluated on 2a without finetuning and 2b with finetuning, and damage maps created. 


