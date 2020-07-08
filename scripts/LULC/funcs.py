

import os #for accessing operating system utilities
from glob import glob #for finding files that match a certain string pattern
import matplotlib.pyplot as plt #for plotting
import numpy as np #for numerical operations
from PIL import Image #for image reading
from joblib import Parallel, delayed #for 'embarassingly parallel' computation - more later!
from numpy.lib.stride_tricks import as_strided as ast #cut an array with the given shape and strides
import random, string #for creating random strings
import tensorflow as tf #tensorflow
import requests #for downloading files from the internet
from skimage.transform import resize #for resizing images
from matplotlib.colors import ListedColormap
from matplotlib import cm

from tensorflow.keras.callbacks import ModelCheckpoint #makes it easy to save the model weights while it trains as a restorable 'checkpoint'
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization #model building layers
from tensorflow.keras.layers import Concatenate, Conv2DTranspose, Flatten, Activation, Add #model building layers
from tensorflow.keras.models import Model #the model construction that binds all the layers together with inputs and outputs

from skimage.filters.rank import median
from skimage.morphology import disk

from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels


# In order to train a neural network to segment the image, we will need to define a image and label batch generator function. Generators are useful because it gets around memory issues associated with the alternative, which is preloading into memory all the images
# 
# The batch generator allows the data to flow to the network in stages efficiently.
# 
# Keras has some limited ability to do this built-in that I encourage you to look at, but eventually you'll run into a need to create your own custom image/label batch generator ...
# 
# so here we go ...
# 
# The function gets called repeatedly by the model duting training, generating batches of images and associated labels by drawing at random from the entire set of validation or train sets (depending on the the context in which it is called, i.e. whether or not it is fed a list of training or validation files).


# In order to train a neural network to segment the image, we will need to define a image and label batch generator function

# The next function is an image batch generator. A `generator` is a special type of python function that `yields` a set of data. In our case, it will yield a set of 8 (because that is the batch size) images and labels drawn randomly from the entire set of `files` provided
def image_batch_generator_flatten(files, N, sz, batch_size = 8):
  
  while True: # this is here because it will be called repeatedly by the training function
    
    #extract a random subset of files of length "batch_size"
    batch = np.random.choice(files, size = batch_size)    
    
    #variables for collecting batches of inputs (x) and outputs (y)
    batch_x = []
    batch_y = []
    
    #cycle through each image in the batch
    for f in batch:
        mask, image = get_pair(f, f.replace('.jpg','.png'), sz)
        batch_x.append(image)
       
        tmp = flatten_labels(mask, labs)
        mask = np.array(tmp == N[0]).astype('int')
        for n in N[1:]:
          mask += np.array(tmp == n).astype('int')
        
        batch_y.append(mask)

    #preprocess a batch of images and masks 
    batch_x = np.array(batch_x) #divide image by 255 to normalize
    batch_y = np.array(batch_y)
    batch_y = np.expand_dims(batch_y,3) #add singleton dimension to batch_y

    yield (batch_x, batch_y) #yield both the image and the label together


def image_batch_generator(files, class_num, sz, batch_size = 8):
  
  while True: # this is here because it will be called repeatedly by the training function
    
    #extract a random subset of files of length "batch_size"
    batch = np.random.choice(files, size = batch_size)    
    
    #variables for collecting batches of inputs (x) and outputs (y)
    batch_x = []
    batch_y = []
    
    #cycle through each image in the batch
    for f in batch:
        # open the image and resize
        image = np.array(Image.open(f).resize(sz))
        batch_x.append(image) #append to our growing list of images

        # open label image by string substitution and resize
        mask = np.array(Image.open(f.replace('.jpg','.png').replace('JPEGImages','SegmentationClass')).resize(sz))
        
        #return 'True' for each pixel equalling the class of interest, then convert that boolean to integer
        mask = (mask==class_num).astype('int') 
        
        batch_y.append(mask) #add to the list of lasbels

    #turn into numpy arrays and expand the dimensions of the label 
    batch_x = np.array(batch_x) 
    batch_y = np.array(batch_y)
    batch_y = np.expand_dims(batch_y,3) #add singleton dimension to batch_y

    yield (batch_x, batch_y) #yield both the image and the label together


# To build and train the model, we'll need to import quite a few `keras` layers

# The following definitions are used to make the residual UNet model, which is what we will be using for our binary segmentation

# We'll talk about this model in more detail later while we wait for it to train. For now, we'll skip over the major details

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
	
# Segmentations of natural landscapes often involve so-called `class imbalance`. This is the situation where the average size of each class is not the same. In other words, different classes tend to occupy different amounts of pixels in imagery, on average. In the presence case, there are fewer pixels associated with smaller objects such as people and man-made objects
# 
# For binary segmentation, class imbalance can be especially bad because the `background` or `everything else` class tends to be large compared to the class of interest
# 
# This is problematic for training if it heavily biases the value of the loss function used to train the model toward the more dominant class
# 
# For these reasons, I recommend using Dice loss which tends to do better for class imbalanced problems. Scores for each class are calculated independently of their relative sizes and hence contribute fairly to the mean score.


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

# the loss is simply 1-coefficient, so its value goes down over time, not up
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# Let's also define another metric. This one might be considered a good allternative for semantic segmentation problems like this, but isn't so effective when the problem is class imbalanced (there are not as many pixels of the class of interest as the background). 
# 
# We'll include it here as an example of passing multiple metrics to the `.optimize()` command so we can keep track of both during model training

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = tf.keras.backend.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou


# The `semantic drone` label imagery are RGB (i.e. 3D), unlike the `aeroscapes` labels, which were 2D. We can't work with the RGB label imagery directly, so we need to `flatten` the labels from 3D to 2D, by converting each vector of unique vector of RGB color codes into a single numeric label

def flatten_labels(im, labs):
   M = []
   for k in range(len(labs)):
      cols = list(labs[k])#[1:]
      msk = ((im[:,:,0]==cols[0])==1) & ((im[:,:,1]==cols[1])==1) & ((im[:,:,2]==cols[2])==1)
      msk = (msk).astype('int')
      M.append(msk)
      del msk

   M2 = [(M[counter]==1)*(1+counter) for counter in range(len(M))]
   msk = np.sum(M2, axis=0)-1
   msk[msk<0] = 0
   return msk


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


class PlotLearning(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('dice_coef'))
        self.val_acc.append(logs.get('val_dice_coef'))
        self.i += 1
        print('i=',self.i,'loss=',logs.get('loss'),'val_loss=',logs.get('val_loss'),
              'dice_coef=',logs.get('dice_coef'),'val_dice_coef=',logs.get('val_dice_coef'),
              'mean_iou=',logs.get('mean_iou'),'val_mean_iou=',logs.get('val_mean_iou'))
        
        #choose a random test image and preprocess
        path = np.random.choice(val_files)
        infile = f'{path}'
        raw = Image.open(infile)

        raw = np.array(raw.resize((512, 512)))

        #predict the mask 
        pred = model.predict(np.expand_dims(raw, 0)).squeeze()
                
        #mask post-processing 
        msk  = (pred>0.5).astype('int')  
      
        msk = 255*np.stack((msk,)*3, axis=-1)
          
        #show the mask and the segmented image 
        combined = np.concatenate([raw, msk, raw* msk], axis = 1)

        plt.axis('off')
        plt.imshow(combined)
        plt.show()      


def build_callbacks(filepath):

    # set checkpoint file 
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                   verbose=0, save_best_only=True, mode='min', 
                                   save_weights_only = True)
        
    # make a list of callback functions
    callbacks = [model_checkpoint, PlotLearning()]

    return callbacks 

# This simple function opens, resizes and returns images and associated labels as numpy arrays

def get_pair2(im, lab, sz):

    label = np.array(Image.open(lab).resize(sz,  Image.NEAREST))
    raw = np.array(Image.open(im).resize(sz,  Image.NEAREST))

    return label, raw



def get_pair(f, class_num, sz):
   image = np.array(Image.open(f).resize(sz))

   # open label image by string substitution and resize
   mask = np.array(Image.open(f.replace('.jpg','.png').replace('JPEGImages','SegmentationClass')).resize(sz))
   
   #return 'True' for each pixel equalling the class of interest, then convert that boolean to integer
   mask = (mask==class_num).astype('int') 

   return image, mask


def crf_labelrefine(input_image, predicted_labels, num_classes):
    
    compat_spat = theta_spat = 1
    compat_col=40 
    theta_col = 40 
    num_iter = 30 
    
    h, w = input_image.shape[:2] #get image dimensions
    
    d = densecrf.DenseCRF2D(w, h, num_classes) #create a CRF object

    # For the predictions, densecrf needs 'unary potentials' which are labels (water or no water)
    predicted_unary = unary_from_labels(predicted_labels.astype('int')+1, num_classes, gt_prob= 0.51)
    
    # set the unary potentials to CRF object
    d.setUnaryEnergy(predicted_unary)

    # to add the color-independent term, where features are the locations only:
    d.addPairwiseGaussian(sxy=(theta_spat, theta_spat), compat=compat_spat, kernel=densecrf.DIAG_KERNEL,
                          normalization=densecrf.NORMALIZE_SYMMETRIC)

    input_image_uint = (input_image * 255).astype(np.uint8) #enfore unsigned 8-bit
    # to add the color-dependent term, i.e. 5-dimensional features are (x,y,r,g,b) based on the input image:    
    d.addPairwiseBilateral(sxy=(theta_col, theta_col), srgb=(5, 5, 5), rgbim=input_image_uint,
                           compat=compat_col, kernel=densecrf.DIAG_KERNEL, 
                           normalization=densecrf.NORMALIZE_SYMMETRIC)

    # Finally, we run inference to obtain the refined predictions:
    refined_predictions = np.array(d.inference(num_iter)).reshape(num_classes, h, w)
    
    # since refined_predictions will be a 2 x width x height array, 
    # each slice respresenting probability of each class (water and no water)
    # therefore we return the argmax over the zeroth dimension to return a mask
    return np.argmax(refined_predictions,axis=0)
