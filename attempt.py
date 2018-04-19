import h5py

from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.models import load_model
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Model

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (105, 105, 1)
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
both = merge([encoded_l,encoded_r],mode=L1_distance,output_shape=lambda x:x[0])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)

optimizer = Adam(0.00006)

siamese_net.compile(loss='binary_crossentropy',optimizer=optimizer)
siamese_net.count_params()


# Data

data_path = '/home/rmn/sit/one-shot/data/omniglot'

with open(os.path.join(data_path, 'train.pickle'), 'rb') as f:
    (X, c) = pickle.load(f)

with open(os.path.join(data_path, 'val.pickle'), 'rb') as f:
    (Xval, cval) = pickle.load(f)

print("Training alphabets: ")
print(c.keys())

print("Validation alphabets: ")
print(cval.keys())


# Siamese Loader
class SiameseLoader:
    def __init__(self, path, data_subsets = ["train", "val"]):
        self.data = {}
        self.categories = {}
        self.info = {}

        for name in data_subsets:
            file_path = os.path.join(path, name + '.pickle')
            print('Loading data from {}'.format(file_path))
            with open(file_path, 'rb') as f:
                (X, c) = pickle.load(f)
                self.data[name] = X
                self.categories[name] = c

    def get_batch(self, batch_size, s='train'):
        X = self.data[s]
        n_classes, n_examples, w, h = X.shape

        # randomly sample several classes to use in the batch
        categories = rng.choice(n_classes, size=(batch_size,), replace=False)
        # initialize two empty arrays for the input image batch
        pairs = [np.zeros((batch_size, h, w,1)) for i in range(2)]
        # initialize vector for the targets, and make one half of it '1's so 2nd half of batch
        # has same class
        targets = np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
            idx_2 = rng.randint(0, n_examples)

            # pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category
            else:
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
            pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h, 1)
        return pairs, targets


    def generate(self, batch_size, s='train'):
        """A generator for batches, so model.fit_generator can be used."""
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)

    
    def make_oneshot_task(self,N,s='val',language=None):
        """Create pairs of test image, support set for testing N way one-shot learning."""
        X = self.data[s]
        n_classes, n_examples, w, h = X.shape
        indices = rng.randint(0,n_examples,size=(N,))
        if language is not None:
            low, high = self.categories[s][language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language,N))
            categories = rng.choice(range(low,high),size=(N,),replace=False)
        else:
            categories = rng.choice(range(n_classes),size=(N,),replace=False)
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
        test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N,w,h,1)
        support_set = X[categories,indices,:,:]
        support_set[0,:,:] = X[true_category,ex2]
        support_set = support_set.reshape(N,w,h,1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets


    def test_oneshot(self, model, N, k, s='val', verbose=0):
        "Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))

        return percent_correct

    
loader = SiameseLoader(data_path)


def concat_images(X):
    """Concatenates a bunch of images into a big matrix for plotting"""
    nc,h,w,_ = X.shape
    X = X.reshape(nc,h,w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w,n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w,y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img

def plot_oneshot_task(pairs):
    """Takes a one-shot task given to a siamese net and """
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.matshow(pairs[0][0].reshape(105,105),cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Example    
# pairs, targets = loader.make_oneshot_task(20,"train","Japanese_(katakana)")
# plot_oneshot_task(pairs)


# Training Loop
print("!")
evaluate_every = 1 # interval for evaluating one-shot tasks
loss_every = 1 # interval for printing loss (iterations)
batch_size = 16
# n_iter = 90000
n_iter = 5000
N_way = 9 # how many classes for testing one-shot tasks
n_val = 500 # how many one-shot tasks to validate on
best = 0
weights_path = os.path.join(data_path,'weigths2')

siamese_net.save('mymodel2.h5')

print("Training")
for i in range(1,n_iter):
    print("---------------------------------------------------")
    (inputs,targets) = loader.get_batch(batch_size)
    loss = siamese_net.train_on_batch(inputs, targets)
    print("Loss : {}".format(loss))
    if i % evaluate_every == 0:
        print("Evaluating")
        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
        if val_acc >= best:
            print("Saving")
            siamese_net.save('mymodel2.h5')
            best = val_acc

    if i % loss_every == 0:
        print("Iteration {}, training loss: {:.2f}".format(i,loss))

    print("Best so far: {}".format(best))
