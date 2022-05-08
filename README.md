# CS6910_Assignment3

## Problem Statement

The problem statement is to use recurrent neural networks to build a transliteration system.

## Installing Libraries

!pip install wandb  (To update experiment values to wandb).\
!pip install transformers (To use the GPT2 model of Q8.).

## Code

We created nine code files for this question.

**Assignment3_Q1.ipynb** :\
Code is written in notebook style. we can download the file and upload to google colab or kaggle and can run all the cells.\
This file contains these following variables which can be manually changed for different combination of parameters of the model which makes the code flexible such that the number_of_en_de, latent_dims, embed_dims, cell_type can be changed.

**Assignment3_Q1_CommandLine.py**:\
This file contains Question 1 code to run it in command line by typing the command

 python Assignment3_Q1_CommandLine.py number_of_en_de latent_dims embed_dims cell_type
 
 Format of command line arguments
    1) number of cells in encoder or decoder (let n)\
    2 to n) hidden states sizes\
    n+1) encoder embedding output size\
    n+2) decoder embedding output size\
    n+3) cell type (rnn or lstm or gru)
 

**Assignment3_Q2.ipynb** :\
This notebook can be uploaded in kaggle or google colab and the cells can be run one after another as in the order to train using sweep parameters and generate the plots and find the test accuracy also.\
train_path = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"\
val_path =   "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"\
test_path = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv" is path set for kaggle.\
To run in colab replace the path accordingly\
Run upto the cell given below to run and plot wandb graphs\
#Run this cell to start sweep\
#Executing sweep
wandb.agent(sweep_id, wandb_sweep, count=30)

**Assignment3_Q4.ipynb** :\
This notebook can be uploaded in kaggle or google colab and the cells can be run one after another as in the order to train using the best model and generate the test accuracy. translate_word() function is used to translate english words into hindi.


**Assignment3_Q5.ipynb** :\
This notebook can be uploaded in kaggle or colab and the cells can be run one after another as in the order to run sweep and find the best parameters with attention and it generates the required plots.

**Assignment3_Q5_Test.ipynb** :\
This notebook can be uploaded in kaggle or colab and the cells can be run one after another as in the order to test the model with the best parameters with attention and it generates the required test accuracy.

**Assignment3-Q6-VisualizationCode.html** "\
This is the required html code to generate the images for Q6. as in folder **A3-Q6-Visualizations**.

**Assignment3_Q8.ipynb** :\
This notebook can be uploaded in kaggle or colab and the cells can be run one after another to generate the song using GPT2 model.


**dl-assignment-3.ipynb** :\
contains initial all the Assignment3 code in one notebook for reference.

## Folders

**A3-Q6-Visualizations** :\
This folder contains all the generated images for Q6.

**predictions_vanilla** :
This folder contains predictions_vanilla.csv which contains corresponding english, predicted hindi and ground truth hindi for all test data without attention

# Files

**predictions_attention.csv** :
This file contains corresponding english, predicted hindi and ground truth hindi for all test data without attention


## Report

The report for this assignment : [link](https://wandb.ai/cs21s045_cs21s011/uncategorized/reports/Assignment-3--VmlldzoxOTQ2MDU5).

## Authors

 - [Prithaj Banerjee](https://github.com/Doeschate)
 - [Kondapalli Jayavardhan](https://github.com/jayavardhankondapalli) 
 
