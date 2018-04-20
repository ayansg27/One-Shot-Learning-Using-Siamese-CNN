import random
import numpy as np
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import binary_crossentropy
from keras.regularizers import l2
from keras import backend as K
from process_faces import SiameseLoader

def imshow(img, text=None, save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75,8,text,style='italic',fontweight='bold',
                 bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

config = {'training_dir'     : './data/faces/training/',
          'testing_dir'      : './data/faces/evaluation/',
          'train_batch_size' : 64,
          'epoch'            : 100}


samples = SiameseLoader()
samples.load_data(config['training_dir'])
samples.load_data(config['testing_dir'],'testing')


# Model
def W_init(shape,name=None):
    values = np.random.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def b_init(shape,name=None):
    values = np.random.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y),axis=1,keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1,shape2 = shapes
    return (shape1[0],1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred,0)))

input_shape = (112, 92, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)

convnet = Sequential()
convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(7,7),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(4,4),activation='relu',
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,
                   kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(4096,activation='sigmoid',kernel_regularizer=l2(1e-3),
                  kernel_initializer=W_init,bias_initializer=b_init))

encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

L1_distance = lambda x : K.abs(x[0]-x[1])
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l,encoded_r])
siamese_net = Model(input=[left_input,right_input],output=distance)

optimizer = Adam(0.00006)

siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)
siamese_net.count_params()


# Training
for epoch in range(config['epoch']):
    img, lbl = samples.random_pair()
    # imgs, lbls = samples.get_batch(config['train_batch_size'])
    target = 1 if lbl[0] == lbl[1] else 0
    img0 = np.reshape(img[0],(112,92,1,1))
    img1 = np.reshape(img[1],(112,92,1,1))
    siamese_net.train_on_batch([img0,img1], target)
    
