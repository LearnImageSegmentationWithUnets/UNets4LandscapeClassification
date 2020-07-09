# UNets4LandscapeClassification
Teaching resources for landscape/use/form image segmentation with U-Nets (a type of deep neural network).

There are two courses that can be accessed and run online through Google Colab

This repository contains versions of those courses that may be adapted for your own purposes, running on the computers you manage and have access to.

----------------------
Step 1: get the code

open an anaconda command window

`git clone --depth 1 https://github.com/MARDAScience/UNets4LandscapeClassification.git`

----------------------
Step 2: create the conda environment

`cd UNets4LandscapeClassification`

`conda env create -f unet_imseg.yml`

`conda activate unet_imseg`

----------------------
Now, you have a few options

1.   Run through the CSDMS workflows (parts 1 and 2) as python scripts

`cd scripts`
`python CSDMS_May2020_part1a_of_2.py`
`python CSDMS_May2020_part1b_of_2.py`
`python CSDMS_May2020_part2_of_2.py`

2.   Run through the CSDMS workflows (parts 1 and 2) as jupyter notebooks

`cd notebooks`
`jupyter notebook`

(open the notebooks from your browser)

3. Train a model on your own data

a. make a new folder inside data and organize your images and label images similarly, into train, test and validation folders
b. make a config file like those for the other data sets
c. copy `CSDMS_May2020_part1a_of_2.py`and adapt it to your needs, modifying (at least) the paths to your data, and other specifics
