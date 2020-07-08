#!/usr/bin/env python
# coding: utf-8

# # Advanced Binary Image Segmentation for the Geo- and Eco-sciences, using Deep Learning
# 
# ## Case Study: Detecting Intertidal Reefs
# 
# #### Daniel Buscombe, MARDA Science
# 
# ![](https://mardascience.com/wp-content/uploads/2019/06/cropped-MardaScience_logo-5.png)

# Before you do anything, go to `File > Save copy in Drive` so you can keep and work on your own copy
# 
# ## Part 3: Train an initial baseline model for detecting intertidal oyster reefs
# 
# This jupyter notebook running on Google Colab is part of the "Advanced Binary Image Segmentation for the Geo- and Eco-sciences, using Deep Learning" course. The main course website can be accessed [here](https://mardascience.gitlab.io/binary_image_segmentation_for_geosciences/#/)
# 
# In the previous tutorial we made functions to read and pair image and label tiles from this data. In this tutorial, we will start by repeating that process without the visualization steps
# 
# Then we will construct, train and evaluate the model. In the [oysterNet paper](https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.134), the authors used a bigger, more sophisticated model for `instance segmentation`, that is, semantic segmentation that is aware of all the different `instances` of the class (i.e. each individual piece of reef). The model they use is called `Mask RCNN`, the implementation of which is [here](https://github.com/matterport/Mask_RCNN). That is a large and very complicated model that is hard to experiment with. The research behind the [oysterNet paper](https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.134) is state-of-the-art.
# 
# Here, we use a simpler model with fewer parameters (namely, a residual UNet) and acheive acceptable results for a semantic segmentation (predicting the masks of where the reefs are, rather than by individual instances). The UNet is the same we used in a [previous set of tutorials](https://mardascience.gitlab.io/deep_learning_landscape_classification/#/) and is relatively simple to adapt and play with to demonstrate a few principles.
# 
# **Actually, in this tutorial we'll start with a basic implementation and get bad results (!) This is on purpose. It demonstrates what you would expect by using an inappropriate loss function for a class imbalanced problem. Class imbalance is probably the norm for binary segmentations, so it's something you need to pay attention to**
# 
# Then in the next parts, we'll adapt our training strategies and try different variations of the model, converging toward a more satisfactory result
# 
# This is designed to demonstrate a problem-solving strategy, and also a principle that often applies to natural imagery:
# 
# > It's not just the model you choose, it's how you train it that counts

# ### Import libraries

# In[ ]:


import os #for accessing operating system utilities
from glob import glob #for finding files that match a certain string pattern
import matplotlib.pyplot as plt #for plotting
import numpy as np #for numerical operations
import random, string #for creating random strings
import tensorflow as tf #tensorflow
import json # for reading lable annotations in json format
import requests #for downloading files 
from PIL import Image, ImageFilter #for reading and filtering imagery
import skimage.draw #for making masks (raster label images) from label annotations
from skimage.transform import resize #for resizing imagery
from psutil import virtual_memory #for interrogating our filesystem and RAM specifications
from imageio import imwrite


# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from random import shuffle
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Conv2DTranspose, Flatten, Activation, Add
from tensorflow.keras.models import Model


# ### Prepare the data

# #### Download the  imagery
# 

# For the remainder of the tutorials, we'll be using this version of the data, consisting of imagery and image labels, with an augmented training set consisting of double the original number of imagery, half of which have been augmented with random flips, zoms and rotations

# In[ ]:


# from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    """
    response = filename for input
    destination = filename for output
    """    
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# In[ ]:


file_id = '19RgkzaD9w-rvAF9uBpqMONnwF0lJVPh3'
destination = 'train_images.tar.gz'
download_file_from_google_drive(file_id, destination)


# In[ ]:


get_ipython().system('tar -xf train_images.tar.gz > tmp.txt')


# In[ ]:


file_id = '1gjWGXO7mtBhJuSOqrG6B1inzLF8TfTSX'
destination = 'val_images.tar.gz'
download_file_from_google_drive(file_id, destination)


# In[ ]:


get_ipython().system('tar -xf val_images.tar.gz > tmp.txt')


# In[ ]:


file_id = '1x8HrgEStkBCdBmFNo1xa0x6LCVE2GE1g'
destination = 'train_labels.tar.gz'
download_file_from_google_drive(file_id, destination)


# In[ ]:


get_ipython().system('tar -xf train_labels.tar.gz > tmp.txt')


# In[ ]:


file_id = '1v0kPgwZVQNrGAk0awJi31W83CYvHJbeX'
destination = 'val_labels.tar.gz'
download_file_from_google_drive(file_id, destination)


# In[ ]:


get_ipython().system('tar -xf val_labels.tar.gz > tmp.txt')


# In[ ]:


root = './content/'


# Check to see how many images we now have to work with

# In[ ]:


train_files = glob(root+"1kx1k_dataset/train_images/data/*.png")
val_files = glob(root+"1kx1k_dataset/val_images/data/*.png")

print("# train files: %i" % (len(train_files)))
print("# validation files: %i" % (len(val_files)))


# ### Setting up model training

# #### Custom batch generator

# We are going to feed images to the network in batches. The batch size (number of images and associated labels) will be ...

# In[ ]:


batch_size = 6


# The next function is an image batch generator. A `generator` is a special type of python function that `yields` a set of data. In our case, it will yield a set of `batch_size` images and labels drawn randomly from the entire set of `files` provided
# 
# It opens the file, reads it into a numpy array with correct dimensions checked, and then does the same for the label image. It scales the image by dividing by 255, do turn the 8-bit data scaled between 0 and 255 to data scaled between 0 and 1. The labels are flattened from 3D (RGB) to 2D (greyscale integers)

# In[ ]:


def image_batch_generator(files, sz, batch_size = 4):

  while True: # this is here because it will be called repeatedly by the training function

    #extract a random subset of files of length "batch_size"
    batch = np.random.choice(files, size = batch_size)

    #variables for collecting batches of inputs (x) and outputs (y)
    batch_x = []
    batch_y = []

    #cycle through each image in the batch
    for f in batch:

        #preprocess the raw images
        raw = Image.open(f)
        raw = raw.resize(sz)
        raw = raw.filter(ImageFilter.UnsharpMask(radius=20, percent=100))
        raw = np.array(raw)

        #check the number of channels because some of the images are RGBA or GRAY
        if len(raw.shape) == 2:
            raw = np.stack((raw,)*3, axis=-1)

        else:
            raw = raw[:,:,0:3]

        #get the image dimensions, find the min dimension, then square the image off
        nx, ny, nz = np.shape(raw)
        n = np.minimum(nx,ny)
        raw = raw[:n,:n,:]

        raw[np.isnan(raw)] = 1e-5 #make any NaNs and InFs a very small number
        raw[np.isinf(raw)] = 1e-5

        batch_x.append(raw)

        #get the masks.
        maskfile = f.replace('_images','_labels').replace('.png','.jpg')
        mask = Image.open(maskfile)
        # the mask is 3-dimensional so get the max in each channel to flatten to 2D
        try:
           mask = np.max(np.array(mask.resize(sz)),axis=2)
        except:
           mask = np.array(mask.resize(sz))

        # water pixels are always greater than 170
        mask = (mask>170).astype('int') ##170 = (2/3)*255

        mask = mask[:n,:n]

        mask[np.isnan(mask)] = 1e-5
        mask[np.isinf(mask)] = 1e-5
        batch_y.append(mask)

    #preprocess a batch of images and masks
    batch_x = np.array(batch_x) #/255. #divide image by 255 to normalize
    batch_y = np.array(batch_y)
    batch_y = np.expand_dims(batch_y,1) #add singleton dimension to batch_y

    yield (batch_x, batch_y) #yield both the image and the label together


# Let's test our generator function
# 
# We will specify input imagery of size `(512, 512, 3)`

# In[ ]:


sz = (512, 512)

gen = image_batch_generator(train_files, sz, batch_size = 4)

images, masks = next(gen)

plt.figure(figsize=(10,10))

for k in range(len(images)):
  plt.subplot(4,2, k+1)
  plt.imshow(images[k])
  plt.imshow(masks[k].squeeze(), vmin=0, vmax=1, alpha=0.25, cmap=plt.cm.Reds)
  plt.colorbar()
  plt.axis('off')


# #### Build the model

# The following definitions are used to make the residual UNet model, which is what we will be using for our binary segmentation

# In[ ]:


def batchnorm_act(x):
    x = BatchNormalization()(x)
    return Activation("relu")(x)

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = batchnorm_act(x)
    return Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

def bottleneck_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    bottleneck = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)
    
    return Add()([conv, bottleneck])

def res_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    bottleneck = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)
    
    return Add()([bottleneck, res])

def upsamp_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    return Concatenate()([u, xskip])

def res_unet(sz, f):
    inputs = Input(sz)
    
    ## downsample  
    e1 = bottleneck_block(inputs, f); f = int(f*2)
    e2 = res_block(e1, f, strides=2); f = int(f*2)
    e3 = res_block(e2, f, strides=2); f = int(f*2)
    e4 = res_block(e3, f, strides=2); f = int(f*2)
    _ = res_block(e4, f, strides=2)
    
    ## bottleneck
    b0 = conv_block(_, f, strides=1)
    _ = conv_block(b0, f, strides=1)
    
    ## upsample
    _ = upsamp_concat_block(_, e4)
    _ = res_block(_, f); f = int(f/2)
    
    _ = upsamp_concat_block(_, e3)
    _ = res_block(_, f); f = int(f/2)
    
    _ = upsamp_concat_block(_, e2)
    _ = res_block(_, f); f = int(f/2)
    
    _ = upsamp_concat_block(_, e1)
    _ = res_block(_, f)
    
    ## classify
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(_)
    
    #model creation 
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
	


# The intersection over union (IoU) metric is a simple metric used to evaluate the performance of a segmentation algorithm. Given two masks $y_{true}, y_{pred}$ we evaluate 
# 
# $$L = \frac{y_{true} \cap y_{pred}}{y_{true} \cup y_{pred}}$$

# In[ ]:


def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = tf.keras.backend.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou


# We use `binary_crossentropy` as the loss function. It is defined:
# 
# $$L = \sum{y_{pred}\log(y_{true}) + (1- y_{pred}) \log(1 -y_{true})}$$
# 

# In[ ]:


model = res_unet(sz+(3,), batch_size)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [mean_iou])


# Let's take a look at how many parameters we have to optimize

# In[ ]:


model.summary()


# We have over 600,000 trainable parameters. This is a lot, for any model, but tiny compared to the number of parameters in some deep learning models, with up to hundreds of millions of trainable parameters

# ### Train the model

# We can use a custom callback function to see how the model is training while it trains
# 
# This must be placed within a class with only one input, `tf.keras.callbacks.Callback`. 
# 
# See https://www.tensorflow.org/guide/keras/custom_callback for more details
# 
# The following will create a plot, at the end of each training epoch, of the segmentation of one validation image using the current state of the model, while it trains
# 
# The plot consists of three images side by side: the validation image, the predicted binary mask, and the masked image showing just that class
# 
# The internal function `on_train_begin` allocates some empty lists that are subsequently filled at the end of the epoch, when `on_epoch_end` is called

# In[ ]:


callbacks = tf.keras.callbacks
backend = tf.keras.backend


class PlotLearning(callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        #self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('dice_coef'))
        self.val_acc.append(logs.get('val_dice_coef'))
        self.i += 1
        print('i=',self.i,'loss=',logs.get('loss'),'val_loss=',logs.get('val_loss'),'mean_iou=',logs.get('mean_iou'),'val_mean_iou=',logs.get('val_mean_iou'))

        #choose a random test image and preprocess
        f = np.random.choice(val_files)

        raw = np.array(Image.open(f).resize(sz))

        #predict the mask
        pred = 255*model.predict(np.expand_dims(raw, 0)).squeeze()
        print(np.max(pred))

        #mask post-processing
        msk  = (pred>200).astype('int') 

        msk = np.stack((msk,)*3, axis=-1)

        #show the mask and the segmented image
        combined = np.concatenate([raw, msk, raw* msk], axis = 1)
        plt.axis('off')
        plt.imshow(combined)
        plt.show()
        #plt.savefig(str(self.i)+'.png', dpi=100, bbox_inches='tight')


# These are parameters that we will pass to callback functions that will adaptively change the pace of training (using an adaptive learning rate), and decide when there is little to be gained from extra training (called 'early stopping')
# 
# Reduce loss on plateau: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
# 
# Early stopping: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
# 
# We'll see that even though the `max_epochs` is set to 100, the model won't need that many epochs
# 

# In[ ]:


# a tolerance for the training. 
min_delta = 0.0001

# minimum learning rate (lambda)
min_lr = 0.0001

# the factor applied to the learning rate when the appropriate triggers are made
factor = 0.2

# number of epochs without performance improvement before the learning rate is adjusted
# and number of epochs with no improvement after which training will be stopped.
patience=25

#number of epochs to wait before resuming normal operation after the learning rate has been reduced.
cooldown=2

max_epochs = 100


# Now we have defined all these hyperparameters (parameters that we chose, not automatically determined by the model), we can make a function that builds all the various callbacks together into a list, which can then be passed to the model training (`.fit()` function)

# In[ ]:


def build_callbacks(filepath, min_delta, min_lr, factor, patience, cooldown):

    earlystop = EarlyStopping(monitor="val_loss", 
                                  mode="min", patience=patience) 
    
    # reduction of learning rate if and when validation scores plateau upon successive epochs
    reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, 
                                    verbose=1, mode='auto', min_delta=min_delta, 
                                    cooldown=cooldown, min_lr=min_lr)

    # set checkpoint file 
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                   verbose=0, save_best_only=True, mode='min', 
                                   save_weights_only = True)
        
    callbacks = [model_checkpoint, reduceloss_plat, earlystop, PlotLearning()]

    return callbacks 


# Next, set up a name for the `.h5` file that will be used to store model weights.
# 
# Finally, we train the model by calling the `.fit()` command and providing all the generators and hyperparameters defined in the callbacks
# 
# The number of training and validation steps is simply the number of respective files divided by the batch size

# In[ ]:


filepath = 'reefs_test_weights_uresnet'+str(batch_size)+'_'+str(max_epochs)+'epochs_iou.h5'

train_generator = image_batch_generator(train_files, sz, batch_size = batch_size)
val_generator  = image_batch_generator(val_files, sz, batch_size = batch_size)
train_steps = len(train_files) //batch_size
val_steps = len(val_files) //batch_size
print(train_steps)
print(val_steps)


# Train the model

# In[ ]:


hist = model.fit(train_generator, 
                epochs = max_epochs, steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath, min_delta, min_lr, factor, patience, cooldown))	


# In the above, we gave an output variable to the `.fit()` command. This contains the training histories. That is, losses and metrics as a function of epoch. You can access the variables in the dictionary like so

# #### Plot the training history

# In[ ]:


hist['history'].keys()


# Let's make a plot of the histories of both train and validation losses and dice coefficients, and also the history of the learning rate

# In[ ]:


plt.figure(figsize=(20,10))
plt.subplot(131)
plt.plot(hist['history']['mean_iou'], 'b', label='train ')
plt.plot(hist['history']['val_mean_iou'], 'k', label='validation ')
plt.xlabel('Epoch number'); plt.ylabel('Mean IOU')
plt.legend()

plt.subplot(132)
plt.plot(hist['history']['loss'], 'b', label='train loss')
plt.plot(hist['history']['val_loss'], 'k', label='validation loss')
plt.xlabel('Epoch number'); plt.ylabel('Loss')
plt.legend()

plt.subplot(133)
plt.plot(hist['history']['lr'], 'g')
plt.xlabel('Epoch number'); plt.ylabel('Learning rate')


# Pretty bad! Why is that?
# 
# We used a metric that isn't very good for class imbalanced problems. This is a situation where the number of reef pixels is far fewer than the non-reef pixels
# 
# This demonstrates - the hard way - why you need to use a loss function that is more suitable for a class-imbalanced binary segmentation
# 

# How do we do better? Let's explore in the next tutorial, when we use a Dice coefficient and Dice loss on the same data and the same model, and this time acheive a much better result

# 
# This jupyter notebook running on Google Colab is part of the "Advanced Binary Image Segmentation for the Geo- and Eco-sciences, using Deep Learning" course. The main course website can be accessed [here](https://mardascience.gitlab.io/binary_image_segmentation_for_geosciences/#/)

# In[ ]:




