import random
import h5py
import numpy as np
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, Flatten, MaxPooling2D, merge
from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import binary_crossentropy
from keras.regularizers import l2
from keras import backend as K
from process_faces import SiameseLoader, DataLoader, show_pair

config = {'training_dir'     : './data/faces/training/',
          'testing_dir'      : './data/faces/evaluation/',
          'train_batch_size' : 64,
          'epoch'            : 500}

samples = SiameseLoader()
samples.load_data(config['training_dir'])
samples.load_data(config['testing_dir'],'testing')


# data_samples = DataLoader('./data/faces')

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


# MODEL
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

# optimizer = Adam(0.00006)
optimizer = Adam(0.0005)

both = merge([encoded_l,encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid',kernel_initializer=W_init,
                   bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)

# siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)

siamese_net.compile(loss='binary_crossentropy',optimizer=optimizer)
siamese_net.count_params()


def test_oneshot(loader, model, N, num_trials, verbose=True):
    '''
    Test average N way one-shot learning accuracy over num_trials one-shot tasks
	N is the number of images of different people in the one-shot task
    '''
    correct_count = 0
    if verbose:
        print('---- Evaluating model on one-shot tasks ...'.format(N))
    for i in range(num_trials):
        inputs, true_class, support_set_classes = loader.make_oneshot_task(N)
        probs = model.predict(inputs)
        if np.argmax(probs) == 0:
            correct_count += 1
    percent_correct = (100.0 * correct_count / num_trials)
    if verbose:
        print('---- Got an average of {}% {} way one-shot learning accuracy'.format(percent_correct,N))
    return percent_correct
    
# Training
N = 9
test_trials = 100
test_every = 1
best = 0
accuracies = []
model_path = '/home/rmn/sit/one-shot/models/faces2.h5'
for epoch in range(config['epoch']):
    print("Epoch : {}".format(epoch))
    inputs, targets = samples.get_training_batch(config['train_batch_size'])
    print("-- Inputs : {}".format(len(inputs[0])))
    loss = siamese_net.train_on_batch(inputs, targets)
    print("-- Loss : {}".format(loss))
    if epoch % test_every == 0:
        print("-- Testing on {} one-shot-tasks".format(test_trials))
        percent_correct = test_oneshot(samples,siamese_net,N,test_trials)
        accuracies.append(percent_correct)
        if percent_correct > best:
            best = percent_correct
            siamese_net.save(model_path)
    print("-- Best so far {}".format(best))
    if epoch > 5:
        print("-- Last 3 rounds {}, {}, {}".format(accuracies[epoch-1],accuracies[epoch-2],accuracies[epoch-3]))


def concat_images(imgs):
    nc,(h,w,_) = len(imgs),imgs[0].shape
    pairs = np.zeros((nc,h,w,1))
    for i in range(nc):
        pairs[i,:,:,:] = imgs[i].reshape(h,w,1)
    pairs.reshape(nc,h,w)

    n = np.ceil(np.sqrt(nc)).astype('int8')
    img = np.zeros((n*h,n*w))
    x,y = 0,0
    for example in range(nc):
        img[x*h:(x+1)*h,y*w:(y+1)*w] = pairs[example].reshape(h,w)
        y+=1
        if y>=n:
            y = 0
            x += 1
    return img

def plot_batch(batch):
    img0 = concat_images(batch[0])
    img1 = concat_images(batch[1])
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.matshow(img0,cmap='gray')
    ax2.matshow(img1,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_oneshot_task(target,support):
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.matshow(target[0].reshape(112,92),cmap='gray')
    img = concat_images(support)
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
