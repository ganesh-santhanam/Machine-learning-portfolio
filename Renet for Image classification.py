
from __future__ import print_function, division
from builtins import range, input

# Let's go up to the end of the first conv block
# to make sure everything has been loaded correctly
# compared to keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input, decode_predictions

from tf_resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock
from tf_resnet_identity_block import IdentityBlock
from tf_resnet_first_layers import ReLULayer, MaxPoolLayer

#  ConvBlock
#  IdentityBlock x 2
#
#  ConvBlock
#  IdentityBlock x 3
#
#  ConvBlock
#  IdentityBlock x 5
#
#  ConvBlock
#  IdentityBlock x 2
#
#  AveragePooling2D
#  Flatten
#  Dense (Softmax)
# ]


# define some additional layers so they have a forward function
class AvgPool:
def __init__(self, ksize):
  self.ksize = ksize

def forward(self, X):
  return tf.nn.avg_pool(
    X,
    ksize=[1, self.ksize, self.ksize, 1],
    strides=[1, 1, 1, 1],
    padding='VALID'
  )

def get_params(self):
  return []

class Flatten:
def forward(self, X):
  return tf.contrib.layers.flatten(X)

def get_params(self):
  return []


def custom_softmax(x):
m = tf.reduce_max(x, 1)
x = x - m
e = tf.exp(x)
return e / tf.reduce_sum(e, -1)


class DenseLayer:
def __init__(self, mi, mo):
  self.W = tf.Variable((np.random.randn(mi, mo) * np.sqrt(2.0 / mi)).astype(np.float32))
  self.b = tf.Variable(np.zeros(mo, dtype=np.float32))

def forward(self, X):
  return tf.matmul(X, self.W) + self.b

def copyFromKerasLayers(self, layer):
  W, b = layer.get_weights()
  op1 = self.W.assign(W)
  op2 = self.b.assign(b)
  self.session.run((op1, op2))

def get_params(self):
  return [self.W, self.b]


class TFResNet:
def __init__(self):
  self.layers = [
    # before conv block
    ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'),
    BatchNormLayer(64),
    ReLULayer(),
    MaxPoolLayer(dim=3),
    # conv block
    ConvBlock(mi=64, fm_sizes=[64, 64, 256], stride=1),
    # identity block x 2
    IdentityBlock(mi=256, fm_sizes=[64, 64, 256]),
    IdentityBlock(mi=256, fm_sizes=[64, 64, 256]),
    # conv block
    ConvBlock(mi=256, fm_sizes=[128, 128, 512], stride=2),
    # identity block x 3
    IdentityBlock(mi=512, fm_sizes=[128, 128, 512]),
    IdentityBlock(mi=512, fm_sizes=[128, 128, 512]),
    IdentityBlock(mi=512, fm_sizes=[128, 128, 512]),
    # conv block
    ConvBlock(mi=512, fm_sizes=[256, 256, 1024], stride=2),
    # identity block x 5
    IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
    IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
    IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
    IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
    IdentityBlock(mi=1024, fm_sizes=[256, 256, 1024]),
    # conv block
    ConvBlock(mi=1024, fm_sizes=[512, 512, 2048], stride=2),
    # identity block x 2
    IdentityBlock(mi=2048, fm_sizes=[512, 512, 2048]),
    IdentityBlock(mi=2048, fm_sizes=[512, 512, 2048]),
    # pool / flatten / dense
    AvgPool(ksize=7),
    Flatten(),
    DenseLayer(mi=2048, mo=1000)
  ]
  self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
  self.output = self.forward(self.input_)

def copyFromKerasLayers(self, layers):
  # conv
  self.layers[0].copyFromKerasLayers(layers[1])
  # bn
  self.layers[1].copyFromKerasLayers(layers[2])
  # cb
  self.layers[4].copyFromKerasLayers(layers[5:17]) # size=12
  # ib x 2
  self.layers[5].copyFromKerasLayers(layers[17:27]) # size=10
  self.layers[6].copyFromKerasLayers(layers[27:37])
  # cb
  self.layers[7].copyFromKerasLayers(layers[37:49])
  # ib x 3
  self.layers[8].copyFromKerasLayers(layers[49:59])
  self.layers[9].copyFromKerasLayers(layers[59:69])
  self.layers[10].copyFromKerasLayers(layers[69:79])
  # cb
  self.layers[11].copyFromKerasLayers(layers[79:91])
  # ib x 5
  self.layers[12].copyFromKerasLayers(layers[91:101])
  self.layers[13].copyFromKerasLayers(layers[101:111])
  self.layers[14].copyFromKerasLayers(layers[111:121])
  self.layers[15].copyFromKerasLayers(layers[121:131])
  self.layers[16].copyFromKerasLayers(layers[131:141])
  # cb
  self.layers[17].copyFromKerasLayers(layers[141:153])
  # ib x 2
  self.layers[18].copyFromKerasLayers(layers[153:163])
  self.layers[19].copyFromKerasLayers(layers[163:173])
  # dense
  self.layers[22].copyFromKerasLayers(layers[175])


def forward(self, X):
  for layer in self.layers:
    X = layer.forward(X)
  return X

def predict(self, X):
  assert(self.session is not None)
  return self.session.run(
    self.output,
    feed_dict={self.input_: X}
  )

def set_session(self, session):
  self.session = session
  for layer in self.layers:
    if isinstance(layer, ConvBlock) or isinstance(layer, IdentityBlock):
      layer.set_session(session)
    else:
      layer.session = session

def get_params(self):
  params = []
  for layer in self.layers:
    params += layer.get_params()


if __name__ == '__main__':
# you can also set weights to None, it doesn't matter
resnet_ = ResNet50(weights='imagenet')

# make a new resnet without the softmax
x = resnet_.layers[-2].output
W, b = resnet_.layers[-1].get_weights()
y = Dense(1000)(x)
resnet = Model(resnet_.input, y)
resnet.layers[-1].set_weights([W, b])

# you can determine the correct layer
# by looking at resnet.layers in the console
partial_model = Model(
  inputs=resnet.input,
  outputs=resnet.layers[175].output
)

# maybe useful when building your model
# to look at the layers you're trying to copy
print(partial_model.summary())

# create an instance of our own model
my_partial_resnet = TFResNet()

# make a fake image
X = np.random.random((1, 224, 224, 3))

# get keras output
keras_output = partial_model.predict(X)

### get my model output ###

# init only the variables in our net
init = tf.variables_initializer(my_partial_resnet.get_params())

# note: starting a new session messes up the Keras model
session = keras.backend.get_session()
my_partial_resnet.set_session(session)
session.run(init)

# first, just make sure we can get any output
first_output = my_partial_resnet.predict(X)
print("first_output.shape:", first_output.shape)

# copy params from Keras model
my_partial_resnet.copyFromKerasLayers(partial_model.layers)

# compare the 2 models
output = my_partial_resnet.predict(X)
diff = np.abs(output - keras_output).sum()
if diff < 1e-10:
  print("Everything's great!")
else:
  print("diff = %s" % diff)
```



```python


from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


# re-size all the images to this
IMAGE_SIZE = [224, 224] # feel free to change depending on dataset

# training config:
epochs = 16
batch_size = 32

# https://www.kaggle.com/paultimothymooney/blood-cells
train_path = '../large_files/blood_cell_images/TRAIN'
valid_path = '../large_files/blood_cell_images/TEST'

# https://www.kaggle.com/moltean/fruits
# train_path = '../large_files/fruits-360/Training'
# valid_path = '../large_files/fruits-360/Validation'
# train_path = '../large_files/fruits-360-small/Training'
# valid_path = '../large_files/fruits-360-small/Validation'

# useful for getting number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(train_path + '/*')


# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()


# add preprocessing layer to the front of VGG
res = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in res.layers:
  layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(res.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)


# create a model object
model = Model(inputs=res.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)



# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)


# test generator to see how it works and some other useful things

# get label mapping for confusion matrix plot later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k

# should be a strangely colored image (due to VGG weights being BGR)
for x, y in test_gen:
  print("min:", x[0].min(), "max:", x[0].max())
  plt.title(labels[np.argmax(y[0])])
  plt.imshow(x[0])
  plt.show()
  break


# create generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)



def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)


# plot some data

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()

from util import plot_confusion_matrix
plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')
