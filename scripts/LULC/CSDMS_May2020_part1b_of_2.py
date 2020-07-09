


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

configfile = 'config_semdrone.json'

with open(os.getcwd()+os.sep+configfile) as f:
    config = json.load(f)

# ### Prepare the "semantic drone" data


train_files = sorted(glob('../../data/semdrone/images/train/*.jpg'))
train_labels = sorted(glob('../../data/semdrone/labels/train/*.png'))
test_files = sorted(glob('../../data/semdrone/images/test/*.jpg'))
test_labels = sorted(glob('../../data/semdrone/labels/test/*.png'))
val_files = sorted(glob('../../data/semdrone/images/val/*.jpg'))
val_labels = sorted(glob('../../data/semdrone/labels/val/*.png'))


train_files = [f.replace('../..',os.getcwd().replace('/scripts/LULC','')) for f in train_files]
test_files = [f.replace('../..',os.getcwd().replace('/scripts/LULC','')) for f in test_files]
val_files = [f.replace('../..',os.getcwd().replace('/scripts/LULC','')) for f in val_files]
train_labels = [f.replace('../..',os.getcwd().replace('/scripts/LULC','')) for f in train_labels]
test_labels = [f.replace('../..',os.getcwd().replace('/scripts/LULC','')) for f in test_labels]
val_labels = [f.replace('../..',os.getcwd().replace('/scripts/LULC','')) for f in val_labels]


print("Number training files: %i" % (len(train_files)))
print("Number validation files: %i" % (len(val_files)))
print("Number testing files: %i" % (len(test_files)))



labs = [k for k in config['classes'].keys()]
cols = [config["classes"][k] for k in labs]

rgb = []
for k in range(len(cols)):
   rgb.append(tuple(int(cols[k][1:][i:i+2], 16) for i in (0, 2, 4)))

class_dict = []

for l,c in zip(labs, rgb):
   class_dict.append((l,)+c)

#
# # Below is a class dictionary, which consists of class names and their RGB colors in the label image
# class_dict = [('unlabeled',	0,	0,	0),
# ('paved-area',	128,	64,	128),
# ('dirt',	130,	76,	0),
# ('grass',	0,	102,	0),
# ('gravel',	112,	103,	87),
# ('water',	28,	42,	168),
# ('rocks',	48,	41,	30),
# ('pool',	0,	50,	89),
# ('vegetation',	107,	142,	35),
# ('roof',	70,	70,	70),
# ('wall',	102,	102,	156),
# ('window',	254,	228,	12),
# ('door',	254,	148,	12),
# ('fence',	190,	153,	153),
# ('fence-pole',	153,	153,	153),
# ('person',	255,	22,	96),
# ('dog',	102,	51,	0),
# ('car',	9,	143,	150),
# ('bicycle',	119,	11,	32),
# ('tree',	51,	51,	0),
# ('bald-tree',	190,	250,	190),
# ('ar-marker',	112,	150,	146),
# ('obstacle',	2,	135,	115),
# ('conflicting',	255,	0,	0)]


# We'll make a separate list only of the class names

labs_txt = [c[0] for c in class_dict]

labs_txt



# Test the function using the first (i.e. `[0]`) image and label pair
label, raw = get_pair2(train_files[0], train_labels[0], (config['sz'], config['sz']))


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
outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1_fig6_semanticdrone_expair.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')

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
outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1_fig7_semanticdrone_expair_flatten.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')

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

gen = image_batch_generator_flatten(train_files, N, (config['sz'], config['sz']), labs, config['batch_size'])
x, y = next(gen)


# which images contain the class(es). 1 means yes
[tmp.max() for tmp in y]


# Plot the batch

plt.figure(figsize=(20,15))

counter = 0
for k in range(config['batch_size']):
  plt.subplot(4,2,counter+1)
  plt.imshow(x[counter],  cmap=plt.cm.gray)
  plt.imshow(y[counter].squeeze(), alpha=0.5, cmap=plt.cm.Greens)
  plt.axis('off')
  counter += 1

outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1_fig8_semanticdrone_exbatch.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')



# Next, set up a name for the `.h5` file that will be used to store model weights.
#
# Finally, we train the model by calling the `.fit()` command and providing all the generators and hyperparameters defined in the callbacks
#
# The number of training and validation steps is simply the number of respective files divided by the batch size

filepath = 'cold_semanticdrone_weights_veg_uresnet'+str(config['batch_size'])+'.h5'

train_generator = image_batch_generator_flatten(train_files, N, (config['sz'], config['sz']), labs, batch_size = config['batch_size'])
val_generator  = image_batch_generator_flatten(val_files, N, (config['sz'], config['sz']), labs, batch_size = config['batch_size'])
train_steps = len(train_files) //config['batch_size']
val_steps = len(val_files) //config['batch_size']
print(train_steps)
print(val_steps)


# Make a new model with the same image size and batch number as before
model = res_unet((config['sz'], config['sz']) + (3,), config['batch_size'])
model.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef, mean_iou])


# ### Train the model with a 'cold start'
#
# The weights are essentially random numbers, so the model has to learn completely from scratch

hist_cold = model.fit(train_generator,
                epochs = config['max_epochs'], steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath))

model_cold = model
del model

# ### Training a model with a 'warm' start
#
# This time we'll make a model and load the aeroscape vegetation weights that we generated earlier over 5 traning epochs. This initializes the model with weights that are more meaningful than random numbers, giving the model a headstart

filepath = 'warm_semanticdrone_weights_veg_uresnet'+str(config['batch_size'])+'.h5'

model = res_unet((config['sz'], config['sz']) + (3,), config['batch_size'])
model.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef, mean_iou])

# transfer learning here:
model.load_weights('aeroscapes_weights_veg_uresnet8.h5')

hist_warm = model.fit(train_generator,
                epochs = config['max_epochs'], steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath))

model_warm = model
del model

# ### Training the model with a 'hot start'
#
# This time, we'll initiate the model with the weights obtained on the aeroscapes dataset over 100 epochs

filepath = 'hot_semanticdrone_weights_veg_uresnet'+str(config['batch_size'])+'.h5'

model = res_unet((config['sz'], config['sz']) + (3,), config['batch_size'])
model.compile(optimizer = 'rmsprop', loss = dice_coef_loss, metrics = [dice_coef, mean_iou])

model.load_weights('aeroscapes_weights_veg_uresnet8.h5')


hist_hot = model.fit(train_generator,
                epochs = config['max_epochs'], steps_per_epoch = train_steps,
                validation_data = val_generator, validation_steps = val_steps,
                callbacks = build_callbacks(filepath))

model_hot = model
del model

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

outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1b_fig9_semanticdrone_transfer_histories.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')


# As you can see, the 'hot' (red) are 'warm' (green) more accurate than the 'cold' (blue) model, with a larger Dice coefficients and smaller loss values. There is marginal advantage oof the hot model over the warm model for this class (vegetation), but that is not true of other classes.
#
# I encourage you to experiment with other classes to see how much variability there is in model performance and hot-starting

# Get the test scores for each model

test_generator = image_batch_generator_flatten(test_files, N, (config['sz'], config['sz']), labs, batch_size = config['batch_size'])

print("# test files: %i" % (len(test_files)))

# some other training parameters
steps = len(test_files) // config['batch_size']


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

for k in range(config['batch_size']):
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

outfile = os.getcwd()+os.sep+'figures'+os.sep+'part1b_fig10_semanticdrone_transfer_res.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close('all')


# Hopefully that gives you a few ideas you can take to your own projects. Be experimental, and get familiar with workflows that can turn your own data and labels into suitable formats for training a model. Unfortunately models are typically trained on relatively small imagery (typically up to 2000 ish square pixels, but more typically less than 1000 square pixels, like here), due to GPU memory limitations

# ### Going Further
#
# I encourage you to go back and train for more epochs, different classes, etc. See what classes work well and which do not (clue: it generally relates to the average number of pixels and frequency of the class within the set). Try with your own data!

# In part 2, we use the UNet model that we used here and use it for multiclass segmentation
