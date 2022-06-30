<img width="100" alt="Cam logo" src= https://seeklogo.com/images/U/university-of-cambridge-logo-E6ED593FBF-seeklogo.com.png >          <img width="100" alt="RMS logo" src=https://www.burstorm.com/wp-content/uploads/RMS-logo-final.png>    

This is the repository for Grace Beaney Colverds AI4ER MRES project titled: 'A deep learning approach to post-hurrciane automatic damage classification'

## 1. Overview

This repository contains all code written for the AI4ER MRES Report.

This project focuses on classifying damage in the aftermath of Hurricanes. This is done by patch based image classification on 

assessing change in the exposure of Caribbean informal settlements over time. This is done firstly by segmenting satellite images to locate informal settlements, and then repeating this process at different times to determine change. Three different methods were used for image segmentation, a Random Forest model as well as two semi-supervised Deep Learning models. This can identify growth or recession of informal settlements. 

Damage detection algorithms were then developed,


This repository is split according to the structure of the write-up, with separate directories for settlement segmentation, change detection, and exposure quantification. Each contain notebooks that can be run to illustrate the different sections of the report.

## 2. Project Structure
```
├── LICENSE
|
├── README.md          <- The top-level README for developers using this project.
|
├── requirements       <- Directory containing the requirement files.
│
├── patch_generation             <- R scripts to read and tile geospatial data for subsequent analysis.
│   │
│   └── preprocessing  <- Scripts to convert raw RGB tiffs into tiled pngs, and to convert shapefiles to geojsons
│
├── src
|   |                  
│   └──          <- Notebook to carry out Bayesian optimisation on the parameters of the ITCfast algorithm
|                         and script to run and evaluate ITCfast with the suggested parameters. 
|
├── models             <- Notebooks to train and test models
|   |                  
│   |── 5conv  <- Notebook to train, test and evaluate a Mask R-CNN model.
|.  | 
|   └-
|   
│
|
└── evaluation    <- Notebooks to test model


