<img width="100" alt="Cam logo" src= https://seeklogo.com/images/U/university-of-cambridge-logo-E6ED593FBF-seeklogo.com.png >          <img width="100" alt="RMS logo" src=https://www.burstorm.com/wp-content/uploads/RMS-logo-final.png>    

# Repository for the Hurricane Damage Classification

## 1. Overview

This repository contains all code written for the AI4ER MRES Report.

This project focuses on classifying damage in the aftermath of Hurricanes. This is done by patch based image classification on 

assessing change in the exposure of Caribbean informal settlements over time. This is done firstly by segmenting satellite images to locate informal settlements, and then repeating this process at different times to determine change. Three different methods were used for image segmentation, a Random Forest model as well as two semi-supervised Deep Learning models. This can identify growth or recession of informal settlements. 

Damage detection algorithms were then developed,


This repository is split according to the structure of the write-up, with separate directories for settlement segmentation, change detection, and exposure quantification. Each contain notebooks that can be run to illustrate the different sections of the report.

## 2. Project Structure
```
├── LICENSE
├── README.md                   <- Main README.
├── settlement_segmentation     <- Settlement segmentation section.
   │
   ├── deepcluster             <- DeepCluster model as well as training and testing notebooks
   │
   ├── liunsupervised          <- Unsupervised feature learning - model building, training, testing          
   |
   └── randomforest            <- RF Classifier training + testing


