import random
import numpy as np
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, Flatten, MaxPooling2D, merge
from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import binary_crossentropy
from keras.regularizers import l2
from keras import backend as K
from process_faces import SiameseLoader, DataLoader, show_pair

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
          'train_batch_size' : 128,
          'epoch'            : 100}

samples = SiameseLoader()
samples.load_data(config['training_dir'])
samples.load_data(config['testing_dir'],'testing')

data_samples = DataLoader('./data/faces')

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
convnet.add(Conv2D(64,(16,16),activation='relu',input_shape=input_shape,
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

# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l,encoded_r])
# siamese_net = Model(input=[left_input,right_input],output=distance)

# optimizer = Adam(lr=0.00006, decay=0.00001)

both = merge([encoded_l,encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid',kernel_initializer=W_init,
                   bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)

# siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)

siamese_net.compile(loss='binary_crossentropy',optimizer=optimizer)
siamese_net.count_params()

# Training
for epoch in range(config['epoch']):
    print("Epoch : {}".format(epoch))
    inputs, targets = samples.batch(config['train_batch_size'])
    inputs = np.array(inputs)
    inputs = inputs.reshape(2,config['train_batch_size'],112,92)
    in_left = inputs[0].reshape(config['train_batch_size'],112,92,1)
    in_right = inputs[1].reshape(config['train_batch_size'],112,92,1)
    inputs = [in_left, in_right]
    targets = [int(e1==e2) for e1,e2 in targets]
    print("-- Inputs : {}".format(len(inputs[0])))
    loss = siamese_net.train_on_batch(inputs, targets)
    print("-- Loss : {}".format(loss))
    
    
# for epoch in range(config['epoch']):
#     print('Epoch : {}'.format(epoch))
#     inputs, targets = data_samples.get_batch(8)
#     print('-- Inputs : {}'.format(len(inputs[0])))
#     loss = siamese_net.train_on_batch(inputs, targets)
#     print('-- Loss : {}'.format(loss))

