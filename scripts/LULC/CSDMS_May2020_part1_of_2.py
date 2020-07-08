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

from funcs import *


# -----------------------------------------------------
# ## Creating a "vegetation detector" 
# 
# The first dataset we will use is called [aeroscapes](https://github.com/ishann/aeroscapes)
# 
# The dataset consists of 3269, 720 x 720 x 3 pixel images and ground-truth masks for 11 classes. 
# 
# The classes are:
# * bckgrnd
# * person
# * bike
# * car
# * drone
# * boat
# * animal
# * obstacle
# * construction
# * vegetation
# * road
# * sky
# 
# The imagery depicts a range of urban and rural landscapes from both oblique nadir viewpoint, acquired at an altitude of 5 to 50 meters above ground from a camera on a UAV (drone). 
# 

# `ls` to list the number of files in the image directory, piped into `wc -l` (literally, word count line) to count the number of images we have in the data set, and then do the same for the labels 

#get_ipython().system('ls aeroscapes/JPEGImages | wc -l')

#get_ipython().system('ls aeroscapes/SegmentationClass/ | wc -l')

#####============================================
#####============================================

# Get a sorted list of images and associated labels using the `glob` function for file pattern recognition 

images = sorted(glob("aeroscapes/JPEGImages/*.jpg"))
labels = sorted(glob("aeroscapes/SegmentationClass/*.png"))

# Before we visualize some examples, let's make a nice color map so we can assign meaningful colors to the label regions corresponding with each class
# 
# We do this by first importing the `ListedColormap` function from matplotlib, that will enable us to make a discrete colormap from a list of colors
# 
# I use the combination of matploltib in-built colors (`r`,`g`,`k`,`c`,`b`,`m`,`y`) and some html color codes (see [here](https://www.w3schools.com/colors/colors_picker.asp))

#####============================================
#####============================================

# We are going to feed images to the network in batches. The batch size (number of images and associated labels) will be ...
batch_size = 8

# The size of the imagery we will use:
sz = (512, 512)

labs = ['bckgrnd','person','bike','car','drone','boat','animal','obstacle','construction','vegetation','road', 'sky']
cols = ['k','#993333','#7589de','b','m','y','#eebfca','#e37711','r','g','#8a8d8d','c']

# These are parameters that we will pass to callback functions that will adaptively change the pace of training (using an adaptive learning rate), and decide when there is little to be gained from extra training (called 'early stopping')
# 
# We set the `max_epochs` to 5, which is very small for training a deep learning model, but we'll run through this quickly to illustrate. In your own time, you should increase the number by increments, to 20, then 50, then 100 or even 200 if you need it. If the loss function has plateaued, the model won't improve any more. Later we'll look into `keras`' in-built [early stopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) functions


max_epochs = 5

# Next, set up a name for the `.h5` file that will be used to store model weights.

filepath = 'aeroscapes_weights_veg_uresnet'+str(batch_size)+'.h5' #construct a filename for the weights file


#####============================================
#####============================================

cmap = ListedColormap(cols)

# We select image and label 1000, open them and plot them using matplotlib
# 
# The bottom layer is a greyscale background image (we used the red or `0th` channel for convenience without converting to intensity)  and the top layer is the label color-coded with out custom `cmap`

label = np.array(Image.open(labels[1000]))
raw = np.array(Image.open(images[1000]))

plt.figure(figsize=(20,20))
plt.imshow(raw[:,:,0], cmap=plt.cm.gray) #plot just the first channel in greyscale
plt.imshow(label, cmap=cmap, alpha=0.5, vmin=0, vmax=11) #plot mask with 50% transparency
cbar = plt.colorbar(shrink=0.5, ticks=np.arange(12)) # make a colorbar, make it smaller and make integer tick spacing
cbar.ax.set_yticklabels(labs)  #tick labels for the colorbar

plt.axis('off')

# Let's take a look at a few more, so we can get a better sense of the scope of the dataset

for k in [0,500,1500,200,2500,3000]:
  label = np.array(Image.open(labels[k]))
  raw = np.array(Image.open(images[k]))

  plt.figure(figsize=(10,10))
  plt.imshow(raw[:,:,0], cmap=plt.cm.gray) #plot just the first channel in greyscale
  plt.imshow(label, cmap=cmap, alpha=0.5, vmin=0, vmax=11) #plot mask with 50% transparency
  cbar = plt.colorbar(shrink=0.5, ticks=np.arange(12))
  cbar.ax.set_yticklabels(labs)  #

  plt.axis('off')


# Ok, we now have a good idea what the imagery and labels look like

# ### Setting up model training

# For this task, we'll choose `vegetation` as our target class. You could choose any class, but I have a specific reason....
# 
# ... after we have trained our model, we are going to use it to give the same model a training headstart to detect vegetation in another similar dataset ... more later

class_num =int(np.where(np.array(labs)=='vegetation')[0])
print(class_num) 

# We can call the generator like so, by first creating a generator object
gen = image_batch_generator(images, class_num, sz, batch_size)

# then using the `next` command to `yield` sample images (`x`) and associated labels (`y`)
x, y = next(gen)

len(y)

# There are `batch_size` number of `x` and `y`. Let's plot them, coloring vegetation green

plt.figure(figsize=(15,15))

counter = 0
for k in range(batch_size):
  plt.subplot(4,4,counter+1)
  plt.imshow(x[counter],  cmap=plt.cm.gray)
  plt.imshow(y[counter].squeeze(), alpha=0.5, cmap=plt.cm.Greens)
  plt.axis('off')
  counter += 1


# Now we are ready to train any model. Let's build one!

# ### Build the model

# Next, we make and compile our model
# 
# As we saw in lesson 1, model compilation is a necessary step, involving specifiying the 'optimizer' (we will use `rmsprop` but `adam` is also a good one to use, in my experience). The loss function is the Dice loss, and the metric we want to keep track of in the dice coefficient
# 
# We will specify input imagery of size `(512, 512, 3)`

# we add (3,) to the tuple 'sz' to tell the model to ue all three image bands (R, G, and B)
model = res_unet(sz + (3,), batch_size)
model.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef, mean_iou]) #'adam'

# ### Save and visualize model

# You can reuse your model a few ways. One is to create the model object using keras layers each time. Another way is to write it out to file, so you can read it back in again
model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)		


# Another useful thing to do is visualize the model architecture. Perhaps the most straight-forward way to do this is print the network diagram to file

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=64
)


# ### Prepare the train, test and validation splits

train_files = []
# open file and read the content in a list
with open('aeroscapes_train_files.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        train_files.append(currentPlace)

#do the same for the test files
test_files = []
with open('aeroscapes_test_files.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        test_files.append(currentPlace)

# and again for the validation files
val_files = []
with open('aeroscapes_val_files.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        val_files.append(currentPlace)


# How many files do we have in each set?

print("Number training files: %i" % (len(train_files)))
print("Number validation files: %i" % (len(val_files)))
print("Number testing files: %i" % (len(test_files)))

# ### Train the model


# Now we have defined all these hyperparameters (parameters that we chose, not automatically determined by the model), we can make a function that builds all the various callbacks together into a list, which can then be passed to the model training (`.fit()` function)
# 
# We are segmenting only vegetation, so we will pass 'class_num' to the batch generator functions because that is what that class appears in the list (counting from zero, as we always do in python)
# 
# Finally, we train the model by calling the `.fit()` command and providing all the generators and hyperparameters defined in the callbacks
# 
# The number of training and validation steps is simply the number of respective files divided by the batch size

train_generator = image_batch_generator(train_files, class_num, sz, batch_size = batch_size) #call the batch generator for train set
val_generator  = image_batch_generator(val_files, class_num, sz, batch_size = batch_size) #call batch generator validation set
train_steps = len(train_files) //batch_size #number of steps per training epoch
val_steps = len(val_files) //batch_size #number of steps per validation epoch
print(train_steps)
print(val_steps)

hist = model.fit(train_generator, 
                epochs = max_epochs, steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath))	

# ### Examining the training history

# In the above, we gave an output variable to the `.fit()` command. This contains the training histories. That is, losses and metrics as a function of epoch. You can access the variables in the dictionary like so

# At this point I would recommend that you save the model weights file, called `aeroscapes_weights_vegetation_uresnet8.h5`, over in the files pane by right-clicking on the file (hit "refresh" if you don't see it) and "download". You will need them if you want to test this workflow on your own computer. Also, if the runtime gets diconnected and you lose what you have done, you can run through the cells until you get to `#model.load_weights(filepath)`, uncomment that, and it will load the weights you saved and resume model training from there

hist.history.keys()

# Let's make a plot of the histories of both train and validation losses and dice coefficients, and also the history of the learning rate. This won't be particularly informative because we only trained for a few epochs, but it will give us an idea  

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


# Everyone will get a different set of curves because we use random subsets of training, testing and validation files 

# ### Test the model

# The most straightfoward way to test the model on all of the test files, is to set up a batch generator like we did for train and validation sets, then use the `model.evaluate()` command to test the imagery in batches.
# 
# This approach doesn't return the segmented imagery, only the scores. It will return the loss value (the model goodness-of-fit), and whatever metrics the model was compiled with. In our case, that was the Dice coefficient and the mean IOU

test_generator = image_batch_generator(test_files, class_num, sz, batch_size = batch_size)

print("# test files: %i" % (len(test_files)))

# some other training parameters
steps = len(test_files) // batch_size

# testing
scores = model.evaluate(test_generator, steps=steps) 

# We could use string formatting to print these numbers to screen

print('loss={loss:0.4f}, Mean Dice={dice_coef:0.4f}'.format(loss=scores[0], dice_coef=scores[1]))


# In order to visualize the segmented imagery, we could generate a batch and use the `model.predict()` function on each sample. So, start by defining a new generator and `yielding` those samples using the `next` command

test_generator = image_batch_generator(test_files, class_num, sz, batch_size = batch_size)
x, y = next(test_generator)


# And make a prediction and plot for each in turn

plt.figure(figsize=(20,20))

for k in range(batch_size):
  plt.subplot(4,4,k+1)
  plt.imshow(x[k])

  #predict the mask 
  pred = model.predict(np.expand_dims(x[k], 0)).squeeze()
        
  # we used the sigmoid function in our classifying layer, so what we get back
  # is a range of values beween 0 and 1 that are like probability scores

  plt.imshow(pred, alpha=0.5, cmap=plt.cm.Greens, vmin=0, vmax=1)
  plt.colorbar()
  plt.axis('off')


# You'll hopefully notice from the above that the model returns a likelihood of the class of interest for each pixel. A conventional way to threshold would be to assign that class to all pixels > 0.5. That is, a hard threshold. But there are other options that we'll explore later that make that deal with these values in a more probabilistic way 
# 
# In the last part of this notebook, I'll demonstrate how you could do something similar with another dataset, but this time you'll give yourself a head-start
# 
# By using the same model architecture, you can simply transfer the model `weights` learned during training, to another model that will subsequently refine those weights for that second data. This general principle is called `transfer learning` (although there are many types beyond this specific one) 
# 
# We'll see that we can 'hot start' a model training in this way, potentially allowing more efficient and quicker training
# 
# ----------------------------------------------------
# ## Transfer learning to a similar data set
# 
# In this part, we are going to take our model weights and use them to train a new model on a similar class in a different data set
# 
# This is one `transfer learning` strategy we could employ.
# 
# The repurposing of models and their pretrained weights - called `transfer learning` is extremely common and encouraged in the computer science community. It allows more efficient training of models, which translates into small model training times and less energy consumption
# 
# Often, there is something generic about the 'features’ that models exploit for classification, so for that reason alone it is often worth carrying out transfer learning
# 

# ### Prepare the "semantic drone" data
# 
# The dataset is available [here](https://www.tugraz.at/index.php?id=22387) and consists of 400 images and associated binary masks of 23 categories. For speed, we will be working with a random subset of these data (100 images and associated masks), but the same workflow would be appropriate for the entire dataset.
# 
# The imagery depicts houses from nadir view acquired at an altitude of 5 to 30 meters above ground. A high resolution camera was used to acquire images at a size of 6000 x 4000 px (24 Mpx). 
# 
# The classes are:
# * unlabeled
# * paved-area
# * dirt
# * grass
# * gravel
# * water
# * rocks
# * pool
# * vegetation
# * roof
# * wall
# * window
# * door
# * fence
# * fence-pole
# * person
# * dog
# * car
# * bicycle
# * tree
# * bald-tree
# * ar-marker
# * obstacle
# * conflicting
# 

# The data files posted on the repository are difficult to access programmatically (i.e. using code), plus we will only work with a subset of the data, put together for the purposes of this lesson. 

# expand the `quarter` folder (it is called quarter because it is a random quarter subset of the entire dataset), we'll see lots of jpg image files and png label files
# 
# Each set of labels is a RGB image

# Perhaps the easiest way to parse all these files is to use the `os.walk` command simply to step through all files in the `quarter` folder. Then we'll create separate image and label lists based on file extension
F = []
for k in os.walk('quarter/'):
   F.append(k)

all_images = sorted(F[0][2])
images = [i for i in all_images if 'jpg' in i]
labels = [i for i in all_images if 'png' in i]


# How many images do we have?

print('%i image files ' % (len(images)))
print('%i label files ' % (len(labels)))


# Below is a class dictionary, which consists of class names and their RGB colors in the label image
class_dict = [('unlabeled',	0,	0,	0),
('paved-area',	128,	64,	128),
('dirt',	130,	76,	0),
('grass',	0,	102,	0),
('gravel',	112,	103,	87),
('water',	28,	42,	168),
('rocks',	48,	41,	30),
('pool',	0,	50,	89),
('vegetation',	107,	142,	35),
('roof',	70,	70,	70),
('wall',	102,	102,	156),
('window',	254,	228,	12),
('door',	254,	148,	12),
('fence',	190,	153,	153),
('fence-pole',	153,	153,	153),
('person',	255,	22,	96),
('dog',	102,	51,	0),
('car',	9,	143,	150),
('bicycle',	119,	11,	32),
('tree',	51,	51,	0),
('bald-tree',	190,	250,	190),
('ar-marker',	112,	150,	146),
('obstacle',	2,	135,	115),
('conflicting',	255,	0,	0)]


# We'll make a separate list only of the class names

labs_txt = [c[0] for c in class_dict]

labs_txt


# The images list needs to be pre-pended with the top level folder name, to be discoverable

images = ['quarter/'+i for i in images]
labels = ['quarter/'+i for i in labels]


# Test the function using the first (i.e. `[0]`) image and label pair
label, raw = get_pair(images[0], labels[0], sz)


# ### Visualize the data
# Make an array of normalized RGB color values and create a color map

rgb = np.array([c[1:] for c in class_dict])/255.

cmap = ListedColormap(rgb,"")


# Make a plot with that defined `cmap`

plt.figure(figsize=(20,20))
plt.imshow(raw[:,:,0], cmap=plt.cm.gray) #plot just the first channel in greyscale
plt.imshow(label, cmap=cmap, alpha=0.5, vmin=0, vmax=len(class_dict)) #plot mask with 50% transparency
cbar = plt.colorbar(shrink=0.5, ticks=np.arange(len(class_dict)+1))
cbar.ax.set_yticklabels(labs_txt) 

plt.axis('off')


# We'll remove the 'unlabeled' category that we don't know what to do with

labs = [c[1:] for c in class_dict]
print(labs)


labs_txt = [l[0] for l in class_dict]

# Let's test our image flattening function 
msk = flatten_labels(label, labs)

# Before we plot, we'll need a colormap. Last time we defined out own colormap with discrete colors. This time I'll show you a different way, if you just want arbitrary colors
cmap = cm.get_cmap('cubehelix', len(labs))

# Now we have a discrete colormap so we can color-code our semi-transparent overlay

plt.figure(figsize=(20,20))
plt.imshow(raw[:,:,0], cmap=plt.cm.gray) #plot just the first channel in greyscale
plt.imshow(msk, cmap=cmap, alpha=0.5, vmin=0, vmax=len(class_dict)) #plot mask with 50% transparency
cbar = plt.colorbar(shrink=0.5, ticks=np.arange(len(class_dict)+1))
cbar.ax.set_yticklabels(labs_txt) 

plt.axis('off')


# ### Preparing for model training

# There are in fact a few 'vegetation' classes in the semantic drone dataset

class_foreground = ['vegetation', 'tree', 'grass'] # a list of the classes we want to predict. The list could alternative contain just one class


# Each of the 3 vegetation classes have a unique code associated with them, so this next bit discovers what those codes are, and makes a list. We'll use that list to make a binary mask of all three vegetation types versus 'background'

N = []

for c in class_foreground:
  class_num = [n for n in range(len(labs_txt)) if labs_txt[n]==c][0]
  print(class_num)
  # you can verify this by 
  print(class_dict[class_num])
  N.append(class_num)


N


# Make a generator instance and get a batch of images (`x`) and labels (`y`), just like before

gen = image_batch_generator_flatten(images, N, sz, batch_size)
x, y = next(gen)


# which images contain the class(es). 1 means yes
[tmp.max() for tmp in y]


# Plot the batch

plt.figure(figsize=(20,15))

counter = 0
for k in range(batch_size):
  plt.subplot(4,2,counter+1)
  plt.imshow(x[counter],  cmap=plt.cm.gray)
  plt.imshow(y[counter].squeeze(), alpha=0.5, cmap=plt.cm.Greens)
  plt.axis('off')
  counter += 1


# ### Prepare the train, test and validation splits

train_files = []
# open file and read the content in a list
with open('semanticdrone_train_files.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        train_files.append(currentPlace)

#do the same for the test files
test_files = []
with open('semanticdrone_test_files.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        test_files.append(currentPlace)

# and again for the validation files
val_files = []
with open('semanticdrone_val_files.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        val_files.append(currentPlace)



print("Number training files: %i" % (len(train_files)))
print("Number validation files: %i" % (len(val_files)))
print("Number testing files: %i" % (len(test_files)))


# Next, set up a name for the `.h5` file that will be used to store model weights.
# 
# Finally, we train the model by calling the `.fit()` command and providing all the generators and hyperparameters defined in the callbacks
# 
# The number of training and validation steps is simply the number of respective files divided by the batch size

filepath = 'cold_semanticdrone_weights_veg_uresnet'+str(batch_size)+'.h5'

train_generator = image_batch_generator(train_files, N, sz, batch_size = batch_size)
val_generator  = image_batch_generator(val_files, N, sz, batch_size = batch_size)
train_steps = len(train_files) //batch_size
val_steps = len(val_files) //batch_size
print(train_steps)
print(val_steps)


# Make a new model with the same image size and batch number as before
model_cold = res_unet(sz + (3,), batch_size)
model_cold.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef, mean_iou])


# ### Train the model with a 'cold start'
# 
# The weights are essentially random numbers, so the model has to learn completely from scratch

hist_cold = model_cold.fit(train_generator, 
                epochs = max_epochs, steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath))	



# ### Training a model with a 'warm' start
# 
# This time we'll make a model and load the aeroscape vegetation weights that we generated earlier over 5 traning epochs. This initializes the model with weights that are more meaningful than random numbers, giving the model a headstart

filepath = 'warm_semanticdrone_weights_veg_uresnet'+str(batch_size)+'.h5'

model_warm = res_unet(sz + (3,), batch_size)
model_warm.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef, mean_iou])

# transfer learning here:
model_warm.load_weights('aeroscapes_weights_veg_uresnet8.h5')

hist_warm = model_warm.fit(train_generator, 
                epochs = max_epochs, steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath))	


# ### Training the model with a 'hot start'
# 
# This time, we'll initiate the model with the weights obtained on the aeroscapes dataset over 100 epochs

filepath = 'hot_semanticdrone_weights_veg_uresnet'+str(batch_size)+'.h5'

model_hot = res_unet(sz + (3,), batch_size)
model_hot.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef, mean_iou])

model_hot.load_weights('aeroscapes_weights_vegetation_uresnet8_100epochs.h5')


hist_hot = model_hot.fit(train_generator, 
                epochs = max_epochs, steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath))	


# ### Comparing the three model training strategies
# 
# Let's make a plot of the histories of both train and validation losses and dice coefficients, and also the history of the learning rate

plt.figure(figsize=(20,10))

## training metrics
plt.subplot(141)
plt.plot(hist_cold.history['dice_coef'], 'b', label='train Dice (cold)')

plt.plot(hist_warm.history['dice_coef'], 'g', label='train Dice (warm)')

plt.plot(hist_hot.history['dice_coef'], 'r', label='train Dice (hot)')

plt.xlabel('Epoch number'); plt.ylabel('Dice coefficent')
plt.legend()

plt.subplot(142)
plt.plot(hist_cold.history['loss'], 'b', label='train loss (cold)')

plt.plot(hist_warm.history['loss'], 'g', label='train loss (warm)')

plt.plot(hist_hot.history['loss'], 'r', label='train loss (hot)')

plt.xlabel('Epoch number'); plt.ylabel('Loss')
plt.legend()

## validation metrics
plt.subplot(143)
plt.plot(hist_cold.history['val_dice_coef'], 'b--', label='validation Dice (cold)')

plt.plot(hist_warm.history['val_dice_coef'], 'g--', label='validation Dice (warm)')

plt.plot(hist_hot.history['val_dice_coef'], 'r--', label='validation Dice (hot)')

plt.xlabel('Epoch number'); plt.ylabel('Dice coefficent')
plt.legend()

plt.subplot(144)
plt.plot(hist_cold.history['val_loss'], 'b--', label='validation loss (cold)')

plt.plot(hist_warm.history['val_loss'], 'g--', label='validation loss (warm)')

plt.plot(hist_hot.history['val_loss'], 'r--', label='validation loss (hot)')

plt.xlabel('Epoch number'); plt.ylabel('Loss')


# As you can see, the 'hot' (red) are 'warm' (green) more accurate than the 'cold' (blue) model, with a larger Dice coefficients and smaller loss values. There is marginal advantage oof the hot model over the warm model for this class (vegetation), but that is not true of other classes. 
# 
# I encourage you to experiment with other classes to see how much variability there is in model performance and hot-starting 

# Get the test scores for each model

test_generator = image_batch_generator_flatten(test_files, N, sz, batch_size = batch_size)

print("# test files: %i" % (len(test_files)))

# some other training parameters
steps = len(test_files) // batch_size


# We'll each have slightly different results because slightly different batches would have been selected during model training, but you should see that the mean Dice score the cold-start model is much lower than the warm and hot start models. On average, the hot start model has the best score

scores = model_cold.evaluate(test_generator, steps=steps) 
print('loss={loss:0.4f}, Mean Dice={dice_coef:0.4f}'.format(loss=scores[0], dice_coef=scores[1]))

scores = model_warm.evaluate(test_generator, steps=steps) 
print('loss={loss:0.4f}, Mean Dice={dice_coef:0.4f}'.format(loss=scores[0], dice_coef=scores[1]))

scores = model_hot.evaluate(test_generator, steps=steps) 
print('loss={loss:0.4f}, Mean Dice={dice_coef:0.4f}'.format(loss=scores[0], dice_coef=scores[1]))


# Finally, we end with some more pretty pictures (outputs from the hot-start model)


plt.figure(figsize=(20,20))

orig_size = (720, 720)

for k in range(batch_size):
  plt.subplot(4,4,k+1)

  resized_image = Image.fromarray(x[k]).resize(orig_size)
  plt.imshow(resized_image)

  #predict the mask 
  # we expand the dimensions (because the model needs a 4d tensor input) then squeeze the 
  # output dimensions down from 3d to 2d
  pred = model_hot.predict(np.expand_dims(x[k], 0)).squeeze()

  resized_label = np.array(Image.fromarray(pred).resize(orig_size))

  #mask post-processing 
  # we used the sigmoid function in our classifying layer, so what we get back
  # is a range of values beween 0 and 1 that are like probability scores
  # I used a threshold of 0.5, but that could be different if you knew different
  msk  = (resized_label>0.5).astype('int')   

  plt.imshow(msk, alpha=0.5, cmap=plt.cm.seismic)
  plt.colorbar(shrink=0.5)
  plt.axis('off')


# Hopefully that gives you a few ideas you can take to your own projects. Be experimental, and get familiar with workflows that can turn your own data and labels into suitable formats for training a model. Unfortunately models are typically trained on relatively small imagery (typically up to 2000 ish square pixels, but more typically less than 1000 square pixels, like here), due to GPU memory limitations

# ### Going Further
# 
# I encourage you to go back and train for more epochs, different classes, etc. See what classes work well and which do not (clue: it generally relates to the average number of pixels and frequency of the class within the set). Try with your own data!

# In part 2, we use the UNet model that we used here and use it for multiclass segmentation
