#!/usr/bin/env python
# coding: utf-8

# # Landcover and Landform Classification with deep neural networks
#
# ## *Virtual* "CSDMS 2020 - Linking Ecosphere and Geosphere". May 20th and 21st, 2020.
#
# -----------------------------
# ### Day 2: using a binary image segmentation model for a multiclass segmentation problem
#
# Before you do anything, go to `File > Save copy in Drive` so you can keep and work on your own copy
#
# In the first part we made a deep convolutional neural network model, specifically a U-Net with residual connections. This is a very useful model framework for landscape scale image segmentation, because it learns both low-level and high-level feature representations of the data, capturing both low- to high-resolution continuum of landscape detail. We saw how well it works on segmenting vegetation in a couple of different aerial (UAV) datasets.
#
# In this part, we will take that model and optimize it for a particular dataset and a multiclass segmentation problem, by combining UNets for individual classes.
#
# I will hopefully convince you of the benefits of a specific approach for segmentation of natural scenery; namely, treating classifications as a series of binary decisions. We'll treat each class separately by considering it against a background of "everything else". That way, we can evaluate each class independently, decide of what classes to use, and we have more options to evaluate what happens in regions predicted to be more than one thing.
#
# We'll utilize a few different python workflows
# * Constructing several UNet models each for a separate class, and loading pre-trained weights to each
# * Using a test set to evaluate the skill of each model/class
# * Resizing predicted segmentations to full image size
# * Post-processing multiclass predictions usig median filters and conditional random fields
#
# All the datasets used in these notebooks are publicly available and explained as they come up in the workflow
#
# -----------------------------------------------
# ### Daniel Buscombe, daniel@mardascience.com, May 2020
#
# ![](https://mardascience.com/wp-content/uploads/2019/06/cropped-MardaScience_logo-5.png)
#

# --------------------------------------------------
# ## Import libraries
#
# Libraries can be imported at any time, but it is often cleaner to define all the libraries we'll need up front.

from funcs import *

# ---------------------------------------------------------------------------
# ## Hyperparameters

# Hi! Let's talk hyperparameters
#
# Hyperparameters are parameters that you decide and include batch size, size of imagery, type of optimizer to use, what loss function to use, etc. Parameters are the weights and biases computed automatically during training. There are thousands to millions of trainable parameters in deep neural networks. Luckily, there are only a few hyperparameters, but the choice of those are critical
#
# We are going to feed images to the network in batches. Batches are images fed into memory for the purposes of training the model. They allow the model to learn in stages - the entire training set is not seen all at once, but in stages, or steps. There are several steps per epoch, a number that is computed as the number of training images divided by the batch size. During each step, the weights of the network are adjusted, so the network trains by nudging weights by small amounts, based on the information in the batch, not all at once, due to information in the entire training set.


configfile = 'config_aeroscapes.json'

with open(os.getcwd()+os.sep+configfile) as f:
    config = json.load(f)


# Before we visualize some examples, let's make a nice color map so we can assign meaningful colors to the label regions corresponding with each class
#
# We do this by first importing the `ListedColormap` function from matplotlib, that will enable us to make a discrete colormap from a list of colors
#
# I use the combination of matploltib in-built colors (`r`,`g`,`k`,`c`,`b`,`m`,`y`) and some html color codes (see [here](https://www.w3schools.com/colors/colors_picker.asp))


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
cmap = ListedColormap(cols)


# If you plan on using colab with these file sets in the future, you could download them to your computer, and reupload them every time you need them

# ## Load the pre-trained weights

#
# We don't have time here to wait for each of the models to train over a sufficient number of epochs, so instead we will pretend we have and use the model weights that I trained earlier. It's the same model and hyperparameters, but trained for each class over 100 epochs
#
# You might think it is inefficient to train the model on different classes, when the model could be generalized to estimate all classes simultaneously. This would work for 'well-posed' problems consisting of strongly differentiated and obvious classes (obvious to both a human labeller and a machine). This is often the case in computer science oriented courses and tutorials that use clean, "anthropocentric" data.
#
# However, the UNet is often relatively quick to train.
#
# Also, and more important, when dealing with natural imagery, it is often a challenge to know what classes are 'tractable'. Binary segmentation is one way to examine the likelihood of the model being able to estimate the class, in isolation from the confounding influence of the other classes. It is much easier, generally, for a machine to make a binary decision rather than a multiclass decision
#
# While multiple binary models might take slightly longer to train, the other advantage is that it allows you to control
#
# Ecologists and geomorphologists are spatially oreinted people, usually already equipped with spatial analysis and other statistical skills that could be put use to decide the optimal way to deal with regions classified as more than one thing. In short, the ambiguity is a nicer problem to have than a 'hard' classification of every pixel
#
# At the end, I propose a way to combine outputs of multiple binary segmentations using Conditional Random Fields, the usage of which is similar to that detailed in [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244)
#


# ## Construct the models

# The below code is an efficient way to make N models based on a list on N classes you have trained weights for
#
# Iterating through a list of classes, we use the `exec` command to execute a set of commands as strings constructed from variables

# In[ ]:

# ## Build the model, prepare the train, test and validation splits

# # we add (3,) to the tuple 'sz' to tell the model to ue all three image bands (R, G, and B)
# model = res_unet(sz + (3,), batch_size)
# model.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef])

sz = (config['sz'], config['sz'])
batch_size = config['batch_size']

for c in ['vegetation', 'bckgrnd', 'construction', 'obstacle', 'road', 'sky']:
  exec('model_'+c+' = res_unet(sz + (3,), batch_size)')
  exec('model_'+c+'.compile(optimizer = "rmsprop", loss = dice_coef_loss, metrics = [dice_coef])')
  exec('model_'+c+'.load_weights("aeroscapes_weights_'+c+'_uresnet8.h5")')


# ## Getting reacquainted with the data

# In[91]:


plt.figure(figsize=(15,15))

counter = 0
for k in range(batch_size):
  plt.subplot(4,4,counter+1)
  infile = np.random.choice(test_files, size = 1)[0]
  x, y = get_pair(infile, 9, sz)
  plt.imshow(x,  cmap=plt.cm.gray)
  plt.imshow(y.squeeze(), alpha=0.5, cmap=plt.cm.Greens)
  plt.axis('off')
  counter += 1

outfile = os.getcwd()+os.sep+'figures'+os.sep+'part2_fig1_imagelabel_exs'+str(k)+'.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')


# Let's remind ourselves of the entire list. We're only using a couple of these
labs


# ## Test the models

# I've commented the code below because it takes a very long time to execute (the test set is very large). But I have also pasted the scores I got when I ran the code

# test_batch_size = 32

# for c in ['vegetation', 'bckgrnd', 'construction', 'obstacle', 'road', 'sky']:

#     class_num =int(np.where(np.array(labs)==c)[0])
#     print(class_num)

#     test_generator = image_batch_generator(test_files, class_num, sz, batch_size = test_batch_size)
#     # some other training parameters
#     steps = len(test_files) // test_batch_size

#     # testing
#     exec('scores = model_'+c+'.evaluate(test_generator, steps=steps)')
#     print('loss={loss:0.4f}, Mean Dice={dice_coef:0.4f}'.format(loss=scores[0], dice_coef=scores[1]))

# 9
# 102/102 [==============================] - 569s 6s/step - loss: 0.0239 - dice_coef: 0.9761
# loss=0.0239, Mean Dice=0.9761
# 0
# 102/102 [==============================] - 576s 6s/step - loss: 0.0546 - dice_coef: 0.9454
# loss=0.0546, Mean Dice=0.9454
# 8
# 102/102 [==============================] - 570s 6s/step - loss: 0.1884 - dice_coef: 0.8116
# loss=0.1884, Mean Dice=0.8116
# 7
# 102/102 [==============================] - 581s 6s/step - loss: 0.3609 - dice_coef: 0.6391
# loss=0.3609, Mean Dice=0.6391
# 10
# 102/102 [==============================] - 581s 6s/step - loss: 0.0181 - dice_coef: 0.9819
# loss=0.0181, Mean Dice=0.9819
# 11
# 102/102 [==============================] - 577s 6s/step - loss: 0.2379 - dice_coef: 0.7621
# loss=0.2379, Mean Dice=0.7621


# The workflow below will predict on a single sample image, for each of the classes, and upscale both the image `x`, ground truth label `y`, and model prediction `pred`, and collect them into lists

#I'm hard-coding an image in so we all get to look at the same thing
infile = train_files[0] # 'aeroscapes/JPEGImages/041003_047.jpg'

# ... but ordinarily you'd probably select it randomly from the list of test files,
# perhaps like this
#infile = np.random.choice(test_files, size = 1)[0]

# the original image size (model training used 512 x 512)
orig_size = (720, 720)

# pre-allocate lists for modelled (M) and true (T) masks
M = []; T = []

counter = 1
for c in ['vegetation', 'bckgrnd', 'construction', 'obstacle', 'road', 'sky']:
    print("Working on %s" % (c))

    class_num =int(np.where(np.array(labs)==c)[0])
    print(class_num)

    x, _ = get_pair(infile, class_num, sz)
    exec('pred = model_'+c+'.predict(np.expand_dims(x, 0)).squeeze()')
    resized_label = (np.array(Image.fromarray(pred).resize(orig_size, resample = Image.NEAREST))>.5)*counter
    M.append(resized_label)

    _, y = get_pair(infile, class_num, sz)
    T.append(((np.array(Image.fromarray(y*255.).resize(orig_size, resample = Image.NEAREST))/255.)>.5)*counter)
    counter += 1


# To get the label matrix we take the sum over the stacking dimension, which happens to be 0. This flattens the matrix down to 2D

M2 = np.sum(np.array(M), axis=0)


# Next we need a new list of labels and assoociated colors

labs_new = ['unknown', 'vegetation', 'bckgrnd', 'construction', 'obstacle', 'road', 'sky']
cols_new = ['k', 'g', '#eebfca','r', 'm','#8a8d8d','c']


# Make a new colormap
cmap_new = ListedColormap(cols_new)


# clip the matrix so pixels classified as more than one thing are classified as zero, or `unknown`
M2[M2>len(labs_new)+1] = 0
M2[M2<0] = 0


# Do the same for the ground truth matrix
T2 = np.sum(np.array(T), axis=0)
T2[T2>len(labs_new)+1] = 0


M2 = M2.astype('int')
T2 = T2.astype('int')


# Check it out

plt.figure(figsize=(20,20))

plt.subplot(121)
plt.imshow(Image.fromarray(x).resize(orig_size))
plt.imshow(M2, alpha=0.4, cmap=cmap_new, vmin=0, vmax=len(labs_new))
plt.axis('off')
plt.title('Ground truth')
cbar = plt.colorbar(shrink=0.25, ticks=np.arange(len(labs_new)+1))
cbar.ax.set_yticklabels(labs_new)

plt.subplot(122)
plt.axis('off')
plt.imshow(Image.fromarray(x).resize(orig_size))
plt.imshow(T2, alpha=0.4, cmap=cmap_new, vmin=0, vmax=len(labs_new)) #plot mask with 40% transparency
plt.title('Model estimate')
cbar = plt.colorbar(shrink=0.25, ticks=np.arange(len(labs_new)+1))
cbar.ax.set_yticklabels(labs_new)

outfile = os.getcwd()+os.sep+'figures'+os.sep+'part2_fig2_compareraw'+str(k)+'.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')

# Hmm, high frequency noise in both ground truth and label, presumably caused by the resizing. Let's use a median filter on the output to smooth over that
#
# `scikit-image` has the tools we need, the median filter and the kernel - we use a disk with radius 15 pixels


# Apply, and visualize

M2 = median(M2, disk(15))
T2 = median(T2, disk(15))

plt.figure(figsize=(20,20))
plt.subplot(121)
plt.axis('off')
plt.imshow(Image.fromarray(x).resize(orig_size))
plt.imshow(T2, alpha=0.4, cmap=cmap_new, vmin=0, vmax=len(labs_new)) #plot mask with 40% transparency
cbar = plt.colorbar(shrink=0.25, ticks=np.arange(len(labs_new)+1))
cbar.ax.set_yticklabels(labs_new)
plt.title('Ground truth')

plt.subplot(122)
plt.axis('off')
plt.imshow(Image.fromarray(x).resize(orig_size))
plt.imshow(M2, alpha=0.4, cmap=cmap_new, vmin=0, vmax=len(labs_new)) #plot mask with 40% transparency
cbar = plt.colorbar(shrink=0.25, ticks=np.arange(len(labs_new)+1))
cbar.ax.set_yticklabels(labs_new)
plt.title('Model estimate')

outfile = os.getcwd()+os.sep+'figures'+os.sep+'part2_fig3_comparefilt'+str(k)+'.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')

# Much better. It is clear the model is as good if not better than the manual label
#
# Let's use the same workflow to look at a few more, randomly selected from the test set

infiles = np.random.choice(test_files, size = 8)

k = 1
for f in infiles:
    print("Working on image %s" % (f))
    M = []; T = []

    counter = 1
    for c in ['vegetation', 'bckgrnd', 'construction', 'obstacle', 'road', 'sky']:
        #print("Working on %s" % (c))
        class_num =int(np.where(np.array(labs)==c)[0])

        x, _ = get_pair(f, class_num, sz)
        exec('pred = model_'+c+'.predict(np.expand_dims(x, 0)).squeeze()')
        resized_label = (np.array(Image.fromarray(pred).resize(orig_size, resample = Image.NEAREST))>.5)*counter
        M.append(resized_label)

        _, y = get_pair(f, class_num, sz)
        T.append(((np.array(Image.fromarray(y*255.).resize(orig_size, resample = Image.NEAREST))/255.)>.5)*counter)
        counter += 1


    M2 = np.sum(np.array(M), axis=0); M2[M2>len(labs_new)+1] = 0; M2[M2<0] = 0
    T2 = np.sum(np.array(T), axis=0); T2[T2>len(labs_new)+1] = 0; T2[T2<0] = 0

    M2 = M2.astype('int')
    T2 = T2.astype('int')

    T2 = median(T2, disk(15))
    M2 = median(M2, disk(15))

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(Image.fromarray(x).resize(orig_size))
    plt.imshow(M2, alpha=0.4, cmap=cmap_new, vmin=0, vmax=len(labs_new))
    plt.axis('off')
    cbar = plt.colorbar(shrink=0.5, ticks=np.arange(len(labs_new)+1))
    cbar.ax.set_yticklabels(labs_new)

    plt.subplot(122)
    plt.axis('off')
    plt.imshow(Image.fromarray(x).resize(orig_size))
    plt.imshow(T2, alpha=0.4, cmap=cmap_new, vmin=0, vmax=len(labs_new)) #plot mask with 40% transparency
    cbar = plt.colorbar(shrink=0.5, ticks=np.arange(len(labs_new)+1))
    cbar.ax.set_yticklabels(labs_new)

    outfile = os.getcwd()+os.sep+'figures'+os.sep+'part2_fig3_comparefilt'+str(k)+'.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close('all')

    k += 1

# Pretty good, overall. How could we do better?

# ## Refining a segmentation using Conditional Random Fields

# Below we use the same task-specific image segmentation algorithm used by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244), called a fully connected Conditional Random Field, or CRF.
#
# Here we'll zero out ambiguous areas (areas classified as more than one thing) and use it to estimate what those labels are, based on a model it will construct that describes the relationship between the the labels present in the label image, and image features (colors and textures), and their relative spatial locations. It's a really powerful model for lots of types of segmentation post-processing tasks. Be guided by your imagination.

# We'll use a CRF model to refine both the UNet model predictions and the labels

# The image and label data are mapped onto nodes of a graph
# nearby grid nodes with similar image intensity, texture and color are likely to be in the same class. The degree of similarity is controlled by the parameter `theta_col` (non-dimensional). As `theta_col` increases, larger image differences are tolerated within the same class.
#
# `compat_col` is a parameter that controls label "compatibility",  by  imposing  a  penalty  for  similar  grid  nodes  that  are  assigned different labels. Similarity in this context is again based on similar image intensity, texture and colors
#
# As `theta_spat` (non-dimensional) increases, larger spatial differences are tolerated within the same class, and `compat_col` imposes a penalty for nearby nodes that are assigned different labels
#
# `num_iter` is the number of iterations the model will take . Increasing this number increases the total inference time
#
# In your own time, play with these parameters and measure the response


# The function will implement the same as above, with the additional step of applying the CRF to both the ground truth and predicted labels

infiles = np.random.choice(test_files, size = 8)
k = 1
for f in infiles:
    print("Working on image %s" % (f))
    M = []; T = []

    counter = 1
    for c in ['vegetation', 'bckgrnd', 'construction', 'obstacle', 'road', 'sky']:
        class_num =int(np.where(np.array(labs)==c)[0])

        x, _ = get_pair(f, class_num, sz)
        exec('pred = model_'+c+'.predict(np.expand_dims(x, 0)).squeeze()')
        resized_label = (np.array(Image.fromarray(pred).resize(orig_size, resample = Image.NEAREST))>.5)*counter
        M.append(resized_label)

        _, y = get_pair(f, class_num, sz)
        T.append(((np.array(Image.fromarray(y*255.).resize(orig_size, resample = Image.NEAREST))/255.)>.5)*counter)
        counter += 1

    try:
        M2 = np.sum(np.array(M), axis=0); M2[M2>len(labs_new)+1] = 0; M2[M2<0] = 0
        T2 = np.sum(np.array(T), axis=0); T2[T2>len(labs_new)+1] = 0; T2[T2<0] = 0

        M2 = M2.astype('int')
        T2 = T2.astype('int')

        T2 = median(T2, disk(15))
        M2 = median(M2, disk(15))

        x = np.array(Image.fromarray(x).resize(orig_size))

        predicted_labels = M2.astype('int')
        predicted_labels[predicted_labels<0] = 0
        M2 = crf_labelrefine(x, predicted_labels, len(labs_new)+1)

        true_labels = T2.astype('int')
        true_labels[true_labels<0] = 0
        M2 = crf_labelrefine(x, true_labels, len(labs_new)+1)

        plt.figure(figsize=(10,10))
        plt.subplot(121)
        plt.imshow(x)
        plt.imshow(M2, alpha=0.4, cmap=cmap_new, vmin=0, vmax=len(labs_new))
        plt.axis('off')
        cbar = plt.colorbar(shrink=0.25, ticks=np.arange(len(labs_new)+1))
        cbar.ax.set_yticklabels(labs_new)

        plt.subplot(122)
        plt.axis('off')
        plt.imshow(x)
        plt.imshow(T2, alpha=0.4, cmap=cmap_new, vmin=0, vmax=len(labs_new)) #plot mask with 40% transparency
        cbar = plt.colorbar(shrink=0.25, ticks=np.arange(len(labs_new)+1))
        cbar.ax.set_yticklabels(labs_new)

        outfile = os.getcwd()+os.sep+'figures'+os.sep+'part2_fig4_comparecrf'+str(k)+'.png'
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close('all')

    except:
        pass
    k += 1

# That is the end! Hope you enjoyed it and found it useful
