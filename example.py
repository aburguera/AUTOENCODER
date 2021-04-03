#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : example
# Description : Simple usage example
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 03-April-2021 - Creation
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

import numpy as np
from autogenerator import AutoGenerator
from automodel import AutoModel
from utils import get_filenames,montage

###############################################################################
# PARAMETERS
###############################################################################

# Paths to the dataset. This example does not use a validation dataset path
# because validation data is taken from the training dataset.
PATH_TRAIN='../../DATA/AUTIMGTR/'       # Path to the training images.
PATH_TEST='../../DATA/AUTIMGTS/'        # Path to the test images

# Expected shape of the images. Please note that the data generator resizes
# the images if requested but it does NOT change the number of channels. So,
# if you have grayscale images mixed with color images you can either pre-pro-
# cess them or modify the AutoGenerator to take that into account.
SHAPE_IMG=(64,64,3)

# Filters to use when creating the autoencoder. See the AutoModel method create
# to understand their meaning.
FILTERS=[16,16,16,64]

###############################################################################
# PREPARE TRAIN DATA
###############################################################################

# Get the train and validation filenames. Validation is 20% of the images (0.2)
# and train is the remaining 80% in the train path.
trainFNames,valFNames=get_filenames(PATH_TRAIN,'png',0.2)

# Create the train and validation data generators. These generators are useful
# to feed the Autoencoder when training and evaluating.
# The train generator randomizes data (to prevent overfitting) but the valida-
# tion generator does not (to provide consistent results).
trainGenerator=AutoGenerator(trainFNames,imgSize=SHAPE_IMG[:2],doRandomize=True)
valGenerator=AutoGenerator(valFNames,imgSize=SHAPE_IMG[:2],doRandomize=False)

###############################################################################
# CREATE AND TRAIN THE MODEL
###############################################################################

# Instantiate an AutoModel object
theModel=AutoModel()

# Create the model
theModel.create(inputShape=SHAPE_IMG,theFilters=FILTERS)

# Train the model just for a few epochs
theModel.fit(x=trainGenerator,validation_data=valGenerator,epochs=10)

###############################################################################
# PREPARE TEST DATA
###############################################################################

# Get the test file names and build the test generator
testFNames=get_filenames(PATH_TEST,'png')
testGenerator=AutoGenerator(testFNames,imgSize=SHAPE_IMG[:2],doRandomize=False)

###############################################################################
# EVALUATE THE MODEL
###############################################################################

theModel.evaluate(testGenerator)

###############################################################################
# SAVE THE MODEL
###############################################################################

theModel.save('TEST')

###############################################################################
# PRINT/PLOT MODEL INFO
###############################################################################

theModel.summary()                          # Print the summary
theModel.plot()                             # Plot the structure
theModel.plot_training_history()            # Training stats
theModel.print_evaluation()                 # Evaluation stats

###############################################################################
# PLOT AN EXAMPLE
###############################################################################

# Get one test batch
[theBatch,_]=testGenerator.__getitem__(0)

# Encode the batch
theFeatures=theModel.encode(theBatch)

# Decode the features
theDecoded=theModel.decode(theFeatures)

# Plot input and decoded together
jointData=np.vstack((theBatch,theDecoded))
montage(jointData)