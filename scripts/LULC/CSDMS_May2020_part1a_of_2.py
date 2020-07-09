#!/usr/bin/env python
# coding: utf-8

# # Landcover and Landform Classification with deep neural networks
#
# ## *Virtual* "CSDMS 2020 - Linking Ecosphere and Geosphere". May 20th and 21st, 2020.
#
# -------------------------------------
# ## Day 1: Constructing a Generic Image Segmentation Model
#
# Before you do anything, go to `File > Save copy in Drive` so you can keep and work on your own copy
#
# By the end of this first part we will have a generically applicable deep convolutional neural network model that can be used for a variety of image segmentation tasks at landscape and smaller scales. The model is based on the popular U-Net model. We'll see how well it works on segmenting vegetation in a couple of different aerial (UAV) datasets.
#
# In this first part, we'll go through a complete generic workflow in `keras` for binary image segmentation (binary in the sense that there are two classes; the class of interest and background) involving custom (and customizable) functions that I've developed. I've found this workflow has been tweakable to suit a number of image segmentation tasks.
#
# We'll utilize a few different python workflows
# * Downloading publicly available datasets from the internet and interrogating their filestructures
# * Plotting imagery and label imagery as a composite, and working with colormaps to visualize the data and classes
# * Making custom image and label batch generators and know how to use them
# * Building a U-Net model with residual layers using `keras`, for binary segmentation
# * Compiling the model with custom loss functions and metrics
# * Splitting the dataset into random train, test and validation splits with varying numbers of images
# * Training the model with custom callback functions
# * Plotting training history variables in a publishable form (i.e. not using tensorboard)
# * Testing the model on a test set and calculating summary statistics
# * Using transfer learning to initialize a model trained on a similar dataset/class, examining the benefits of 'warm starting' a model, which is transfering the weights of a model trained one one dataset to initiate the training of another
#
# -------------------------------------
# In the next and final part, we will take that model and optimize it for a particular dataset and a multiclass segmentation problem, where we'll combine models trained on every class of interest.
#
# I will also stress that there are many deep learning models that do essentially the same thing, but getting them to work for you and your real-world messy data can be tricky. Along the way, I will hopefully convince you of the benefits of a specific approach for segmentation of natural scenery; namely, treating classifications as a series of binary decisions. We'll treat each class separately by considering it against a background of "everything else". That way, we can evaluate each class independently, decide of what classes to use, and we have more options to evaluate what happens in regions predicted to be more than one thing.
#
# All the datasets used in these notebooks are publicly available and explained as they come up in the workflow
#
# -------------------------------------
# ### Daniel Buscombe, daniel@mardascience.com, May 2020
#
# ![](https://mardascience.com/wp-content/uploads/2019/06/cropped-MardaScience_logo-5.png)
#

# ------------------------------------------------
# ## Import libraries
#
# Libraries can be imported at any time, but it is often cleaner to define all the libraries we'll need up front.
#
#
# Tensorflow is a library that allows the deployment of fast machine learning solutions on various platforms - CPU, GPU, mobile devices, etc.
#
# It is written in C++ for speed, and wrapped in python for usability. It has its own low-level API, and also a high-level API called Keras. Keras allows construction of complex Tensorflow code using more friendly and simple syntax.

global val_files

from funcs import *

def build_callbacks(filepath):

    # set checkpoint file
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                   verbose=0, save_best_only=True, mode='min',
                                   save_weights_only = True)

    # make a list of callback functions
    callbacks = [model_checkpoint, PlotLearning()]

    return callbacks

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
        #plt.show()
        plt.savefig("epoch"+str(epoch)+".png", dpi=300)
        plt.close("all")

# -----------------------------------------------------
# ## Creating a "vegetation detector"
#
# The first dataset we will use is called [aeroscapes]()

# Before we visualize some examples, let's make a nice color map so we can assign meaningful colors to the label regions corresponding with each class
#
# We do this by first importing the `ListedColormap` function from matplotlib, that will enable us to make a discrete colormap from a list of colors
#
# I use the combination of matploltib in-built colors (`r`,`g`,`k`,`c`,`b`,`m`,`y`) and some html color codes (see [here](https://www.w3schools.com/colors/colors_picker.asp))

# These are parameters that we will pass to callback functions that will adaptively change the pace
# of training (using an adaptive learning rate),
# and decide when there is little to be gained from extra training (called 'early stopping')
#

configfile = 'config_aeroscapes.json'

with open(os.getcwd()+os.sep+configfile) as f:
    config = json.load(f)


filepath = config['dataset']+'_weights_veg_uresnet'+str(config['batch_size'])+'.h5' #construct a filename for the weights file


##NEED TO MAKE SPLITS AS DIRECTORIES
train_files = sorted(glob('../../data/aeroscapes/JPEGImages/train/*.jpg'))
test_files = sorted(glob('../../data/aeroscapes/JPEGImages/test/*.jpg'))
val_files = sorted(glob('../../data/aeroscapes/JPEGImages/val/*.jpg'))

train_labels = sorted(glob('../../data/aeroscapes/SegmentationClass/train/*.png'))
test_labels = sorted(glob('../../data/aeroscapes/SegmentationClass/test/*.png'))
val_labels = sorted(glob('../../data/aeroscapes/SegmentationClass/val/*.png'))


# How many files do we have in each set?

print("Number training files: %i" % (len(train_files)))
print("Number validation files: %i" % (len(val_files)))
print("Number testing files: %i" % (len(test_files)))


labs = [k for k in config['classes'].keys()]
cols = [config["classes"][k] for k in labs]

#####============================================
#####============================================

cmap = ListedColormap(cols)

# We select image and label 1000, open them and plot them using matplotlib
#
# The bottom layer is a greyscale background image (we used the red or `0th` channel
# for convenience without converting to intensity)
# and the top layer is the label color-coded with out custom `cmap`

label = np.array(Image.open(train_labels[100]))
raw = np.array(Image.open(train_files[100]))

plt.figure(figsize=(20,20))
plt.imshow(raw[:,:,0], cmap=plt.cm.gray) #plot just the first channel in greyscale
plt.imshow(label, cmap=cmap, alpha=0.5, vmin=0, vmax=11) #plot mask with 50% transparency
cbar = plt.colorbar(shrink=0.5, ticks=np.arange(12)) # make a colorbar, make it smaller and make integer tick spacing
cbar.ax.set_yticklabels(labs)  #tick labels for the colorbar

plt.axis('off')
outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1_fig1_imagelabel_ex.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')

# Let's take a look at a few more, so we can get a better sense of the scope of the dataset

for k in [0,100,200,300,400,450]:
  label = np.array(Image.open(train_labels[k]))
  raw = np.array(Image.open(train_files[k]))

  plt.figure(figsize=(10,10))
  plt.imshow(raw[:,:,0], cmap=plt.cm.gray) #plot just the first channel in greyscale
  plt.imshow(label, cmap=cmap, alpha=0.5, vmin=0, vmax=11) #plot mask with 50% transparency
  cbar = plt.colorbar(shrink=0.5, ticks=np.arange(12))
  cbar.ax.set_yticklabels(labs)  #

  plt.axis('off')

  outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1_fig2_imagelabel_exs'+str(k)+'.png'
  plt.savefig(outfile, dpi=300, bbox_inches='tight')
  plt.close('all')

# Ok, we now have a good idea what the imagery and labels look like

# ### Setting up model training

# For this task, we'll choose `vegetation` as our target class.
#You could choose any class, but I have a specific reason....
#
# ... after we have trained our model, we are going to use it to give the
# same model a training headstart to detect vegetation in another similar dataset ... more later

class_num =int(np.where(np.array(labs)=='vegetation')[0])
print(class_num)

# We can call the generator like so, by first creating a generator object
gen = image_batch_generator(train_files, class_num, (config['sz'], config['sz']), config['batch_size'])

# then using the `next` command to `yield` sample images (`x`) and associated labels (`y`)
x, y = next(gen)

len(y)

# There are `batch_size` number of `x` and `y`. Let's plot them, coloring vegetation green

plt.figure(figsize=(15,15))

counter = 0
for k in range(config['batch_size']):
  plt.subplot(4,4,counter+1)
  plt.imshow(x[counter],  cmap=plt.cm.gray)
  plt.imshow(y[counter].squeeze(), alpha=0.5, cmap=plt.cm.Greens)
  plt.axis('off')
  counter += 1

outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1_fig3_imagelabel_batch.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')

# Now we are ready to train any model. Let's build one!

# ### Build the model

# Next, we make and compile our model
#
# As we saw in lesson 1, model compilation is a necessary step,
#involving specifiying the 'optimizer' (we will use `rmsprop`
#but `adam` is also a good one to use, in my experience).

#The loss function is the Dice loss, and the metric we want to keep track of in the dice coefficient
#
# We will specify input imagery of size `(512, 512, 3)`

# we add (3,) to the tuple 'sz' to tell the model to ue all three image bands (R, G, and B)
model = res_unet((config['sz'], config['sz']) + (3,), config['batch_size'])
model.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef, mean_iou]) #'adam'

# ### Save and visualize model

# You can reuse your model a few ways.
#One is to create the model object using keras layers each time.
#Another way is to write it out to file, so you can read it back in again
# model_json = model.to_json()
# with open('model.json', "w") as json_file:
#     json_file.write(model_json)


# Another useful thing to do is visualize the model architecture.
#Perhaps the most straight-forward way to do this is print the network diagram to file

# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=64
# )

# ### Train the model


# Now we have defined all these hyperparameters (parameters that we chose,
# not automatically determined by the model),
# we can make a function that builds all the various callbacks together into a list,
#which can then be passed to the model training (`.fit()` function)
#
# We are segmenting only vegetation, so we will pass 'class_num' to the
#batch generator functions because that is what that class appears in the
#list (counting from zero, as we always do in python)
#
# Finally, we train the model by calling the `.fit()` command and providing
#all the generators and hyperparameters defined in the callbacks
#
# The number of training and validation steps is simply the number of
#respective files divided by the batch size

train_generator = image_batch_generator(train_files, class_num, (config['sz'], config['sz']), batch_size = config['batch_size'])
#call the batch generator for train set
val_generator  = image_batch_generator(val_files, class_num, (config['sz'], config['sz']), batch_size = config['batch_size'])
#call batch generator validation set
train_steps = len(train_files) //config['batch_size'] #number of steps per training epoch
val_steps = len(val_files) //config['batch_size'] #number of steps per validation epoch
print(train_steps)
print(val_steps)

hist = model.fit(train_generator,
                epochs = config['max_epochs'], steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath))

# ### Examining the training history

# In the above, we gave an output variable to the `.fit()` command.
#This contains the training histories. That is, losses and metrics as a function of epoch.
#You can access the variables in the dictionary like so

# At this point I would recommend that you save the model weights file,
#called `aeroscapes_weights_vegetation_uresnet8.h5`,
#over in the files pane by right-clicking on the file (hit "refresh" if you don't see it) and "download".
#You will need them if you want to test this workflow on your own computer.
#Also, if the runtime gets diconnected and you lose what you have done,
#you can run through the cells until you get to `#model.load_weights(filepath)`,
#uncomment that, and it will load the weights you saved and resume model training from there

hist.history.keys()

# Let's make a plot of the histories of both train and validation losses and
#dice coefficients, and also the history of the learning rate.
#This won't be particularly informative because we only trained for a few epochs, but it will give us an idea

plt.figure(figsize=(20,10))
plt.subplot(131)
plt.plot(hist.history['dice_coef'], 'b--', label='train Dice coefficient')
plt.plot(hist.history['val_dice_coef'], 'k', label='validation Dice coefficient')
plt.xlabel('Epoch number'); plt.ylabel('Dice coefficent')
plt.legend()

plt.subplot(132)
plt.plot(hist.history['loss'], 'b--', label='train loss')
plt.plot(hist.history['val_loss'], 'k', label='validation loss')
plt.xlabel('Epoch number'); plt.ylabel('Loss')
plt.legend()

plt.subplot(133)
plt.plot(hist.history['dice_coef'],hist.history['mean_iou'], 'k-s')
plt.plot(hist.history['val_dice_coef'],hist.history['val_mean_iou'], 'b--o')
plt.xlabel('Dice coefficient'); plt.ylabel('Mean IOU')

outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1_fig4_aeroscapes_model_training.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')

# Everyone will get a different set of curves because we use random subsets of training,
#testing and validation files

# ### Test the model

# The most straightfoward way to test the model on all of the test files,
#is to set up a batch generator like we did for train and validation sets,
#then use the `model.evaluate()` command to test the imagery in batches.
#
# This approach doesn't return the segmented imagery, only the scores.
#It will return the loss value (the model goodness-of-fit), and whatever metrics
#the model was compiled with. In our case, that was the Dice coefficient and the mean IOU

test_generator = image_batch_generator(test_files, class_num, (config['sz'], config['sz']), batch_size = config['batch_size'])

print("# test files: %i" % (len(test_files)))

# some other training parameters
steps = len(test_files) // config['batch_size']

# testing
scores = model.evaluate(test_generator, steps=steps)

# We could use string formatting to print these numbers to screen

print('loss={loss:0.4f}, Mean Dice={dice_coef:0.4f}'.format(loss=scores[0], dice_coef=scores[1]))


# In order to visualize the segmented imagery, we could generate a batch and use the `model.predict()`
#function on each sample. So, start by defining a new generator and `yielding` those
#samples using the `next` command

test_generator = image_batch_generator(test_files, class_num, (config['sz'], config['sz']), batch_size = config['batch_size'])
x, y = next(test_generator)


# And make a prediction and plot for each in turn

plt.figure(figsize=(20,20))

for k in range(config['batch_size']):
  plt.subplot(4,4,k+1)
  plt.imshow(x[k])

  #predict the mask
  pred = model.predict(np.expand_dims(x[k], 0)).squeeze()

  # we used the sigmoid function in our classifying layer, so what we get back
  # is a range of values beween 0 and 1 that are like probability scores

  plt.imshow(pred, alpha=0.5, cmap=plt.cm.Greens, vmin=0, vmax=1)
  plt.colorbar()
  plt.axis('off')


outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1_fig5_aeroscapes_model_test.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')


# You'll hopefully notice from the above that the model returns a likelihood of the
#class of interest for each pixel.
#A conventional way to threshold would be to assign that class to all pixels > 0.5.
# That is, a hard threshold. But there are other options that we'll explore later that make
#that deal with these values in a more probabilistic way
#
# In the last part of this notebook, I'll demonstrate how you could do something similar with another dataset,
#but this time you'll give yourself a head-start
#
# By using the same model architecture, you can simply transfer the model `weights` learned during training,
#to another model that will subsequently refine those weights for that second data.
#This general principle is called `transfer learning` (although there are many types beyond this specific one)
#
# We'll see that we can 'hot start' a model training in this way, potentially allowing more efficient and
#quicker training
