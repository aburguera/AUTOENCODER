#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : main
# Description : Trains, evaluates, saves and plots some autoencoder stats.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 02-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# SET GPU
###############################################################################

# These must be the first lines to execute. Restart the kernel before.
# If you don't have GPU/CUDA or this does not work, just set it to False or
# (preferrably) remove these two lines.
from utils import set_gpu
set_gpu(True)

###############################################################################
# IMPORTS
###############################################################################

from autogenerator import AutoGenerator
from autotools import autobuild,autoshow
from utils import get_filenames

###############################################################################
# PARAMETERS
###############################################################################

# Model input shape.
MODEL_INPUT_SHAPE=(64,64,3)

# List of model filters to train/evalyate. See AutoModel to understand how
# these filtera are used. The latent space is of size
# ((64/(2**len(filter)))**2)*filter[-1]
MODEL_FILTERS=[[128,128,16],    # Size of latent space = 1024
               [128,4],         # Size of latent space = 1024
               [128,8],         # Size of latent space = 2048
               [128,128,32],    # Size of latent space = 2048
               [128,16],        # Size of latent space = 4096
               [128,128,64],    # Size of latent space = 4096
               [128,32],        # Size of latent space = 8192
               [128,128,128]]   # Size of latent space = 8192

# Number of epochs to train
MODEL_EPOCHS=100

# Paths
PATH_TRAIN='../../DATA/AUTIMGTR/'   # Train images path
PATH_TEST='../../DATA/AUTIMGTS/'    # Test images path
PATH_MODELS='../../DATA/MODELS/'    # Model storage path

# File extensions
EXT_IMAGE='png'                     # Image file extension

# Split percentages
SPLIT_VAL=0.2                       # Ratio of train images to use as validat.

###############################################################################
# TRAIN, EVALUATE AND SAVE
###############################################################################

# Train, evaluate and save autoencoders using all the filters in MODEL_FILTERS
allNames=[]
for curFilters in MODEL_FILTERS:
    baseName=autobuild(PATH_TRAIN,PATH_TEST,PATH_MODELS,EXT_IMAGE,MODEL_INPUT_SHAPE,curFilters,MODEL_EPOCHS,SPLIT_VAL)
    allNames.append(baseName)

###############################################################################
# SHOW MODELS STATS
# Please, note that this is too verbose. The code is here just as an example of
# the available stats and how to access it.
###############################################################################

# Create a data generator from test images and pick one batch just to be used
# as an example of encoding and decoding images.
testGenerator=AutoGenerator(get_filenames(PATH_TEST,EXT_IMAGE),imgSize=MODEL_INPUT_SHAPE[:2],doRandomize=False)
[testBatch,_]=testGenerator.__getitem__(0)

# Show stats.
for curName in allNames:
    autoshow(curName,testBatch)