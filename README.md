## DEEMD
This repository provides training and testing scripts for the paper DEEMD: Drug Efficacy Estimation against SARS-CoV-2 based on cell Morphology with Deep multiple instance learning.
## DEEMD: Drug Efficacy Estimation against SARS-CoV-2 based on cell Morphology with Deep multiple instance learning
Drug repurposing can accelerate the identification of effective compounds for clinical use against SARS-CoV-2, with the advantage of pre-existing clinical safety data and an established supply chain. RNA viruses such as SARS-CoV-2 manipulate cellular pathways and induce reorganization of subcellular structures to support their life cycle. These morphological changes can be quantified using bioimaging techniques. In this work, we developed DEEMD: a computational pipeline using deep neural network models within a multiple instance learning (MIL) framework, to identify putative treatments effective against SARS-CoV-2 based on morphological analysis of the publicly available RxRx19a dataset. This dataset consists of fluorescence microscopy images of SARS-CoV-2 non-infected cells and infected cells, with and without drug treatment. DEEMD first extracts discriminative morphological features to generate cell morphological profiles from the non-infected and infected cells. These morphological profiles are then used in a statistical model to estimate the applied treatment efficacy on infected cells based on similarities to non-infected cells. DEEMD is capable of localizing infected cells via weak supervision without any expensive pixel-level annotations. DEEMD identifies known SARS-CoV-2 inhibitors, such as *Remdesivir* and *Aloxistatin*, supporting the validity of our approach. DEEMD is scalable to process and screen thousands of treatments in parallel and can be explored for other emerging viruses and datasets to rapidly identify candidate antiviral treatments in the future.


## How to use RxRx19a dataset for MIL
Get the images and the meta data from [Recursion website](https://www.rxrx.ai/rxrx19a). Include a column named `path` into the meta dataframe which contains the relative path to the images files. Use this csv file along with parameter **base_path** to properly load the dataset into the dataloader. 

## Adapting DEEMD to a new dataset
To transfer DEEMD pipeline to another dataset follow the guideline below. This guide assumes that you are working with a dataset of fluorescence microscopy images for drug repurposing, however, the deep multiple instance learning framework can be applied to other problems as well.

## Preprocessing the dataset
1. Create a csv file that includes meta data of the samples in the dataset. In particular, this should include a `path` column for loading the samples.
2. Split the csv file into train, validation, and test set.
3. Use the training set to calculate the emperical mean and standard deviation for each channel of the samples.
4. Update `Training/SMIL.py` with calculated channel mean and standard deviations.

## Training with Deep MIL
1. For training the network based on a new dataset, modify the data loader to properly accomodate your needs. This mostly consists of setting the number of input channels in the sample images, the transfomations required to be performed on the images, meta data required to be stored for each instance, and the patch selection mechanism. 
2. Define a deep neural network along with the loss function suitable for the task at hand.
3. Then run the `Training/SMIL.py` which trains a model based on ResNet50 and logs the training process in `convergence.csv`. The best model based on the validation set, if presented, is stored at `checkpoint_best.pth`.

## Inference with Deep MIL
Provide proper arguments for `Analysis/analyze.py` and run it. It will provide you with the predicted infection probabilities for each patch in `h5` format, estimated infection probabilities in `.csv` which is later used for calculating treatment efficacy scores. It will also generates performance report for each split provided.

## Treatment efficacy score estimation
Use the `.csv` file generated in inference to run the `Analysis/Analysis.R`. It will generate a ranked list of treatments along with their estimated efficacy score in `.csv`.






