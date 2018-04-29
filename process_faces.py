import os
import sys
import pickle
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import numpy as np
import random

train_folder = '/home/rmn/sit/one-shot/data/faces/training'
eval_folder = '/home/rmn/sit/one-shot/data/faces/evaluation'

save_path = '/home/rmn/sit/one-shot/data/faces'

f_dict = {}

def load_faces(path,n=0):
    X = []
    y = []
    cat_dict = {}
    faces_dict = {}
    curr_y = n

    for person_id in os.listdir(path):
        print('Loading person ID: {}'.format(person_id))
        faces_dict[person_id] = [curr_y,None]
        person_path = os.path.join(path,person_id)
        category_images = []

        for person_image in os.listdir(person_path):
            cat_dict[curr_y] = (person_id,person_image)
            # category_images = []
            image_path = os.path.join(person_path,person_image)
            image = imread(image_path)
            print('-- Image:{}'.format(person_image))
            category_images.append(image)
            y.append(curr_y)
            print("len(category_images):{}".format(len(category_images)))
        try:
            X.append(np.stack(category_images))
        except ValueError as e:
            print(e)
            print('Error -- Category Images:{}'.format(category_images))
        curr_y += 1
        faces_dict[person_id][1] = curr_y - 1

    y = np.vstack(y)
    X = np.stack(X)
    return X, y, faces_dict

def pickle_data(save_path,data='training'):
    if data is 'training':
        folder = train_folder
    else:
        folder = eval_folder
    X,y,c = load_faces(folder)
    with open(os.path.join(save_path,data+'.pickle'),'wb') as f:
        pickle.dump((X,c),f)

class DataLoader(object):
    def __init__(self,path,data_subsets=['training','evaluation']):
        self.data = {}
        self.categories = {}
        self.info = {}

        for name in data_subsets:
            file_path = os.path.join(path, name + '.pickle')
            print('Loading data from {}'.format(file_path))
            with open(file_path, 'rb') as f:
                (X,c) = pickle.load(f)
                self.data[name] = X
                self.categories[name] = c

    def get_batch(self,batch_size,s='training'):
        X = self.data[s]
        n_classes, n_examples, w, h = X.shape

        categories = np.random.choice(n_classes, size=(batch_size,), replace=False)
        pairs = [np.zeros((batch_size,w,h,1)) for i in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = np.random.randint(0,n_examples)
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w,h,1)
            idx_2 = np.random.randint(0,n_examples)

            if i >= batch_size//2:
                category_2 = category
            else:
                category_2 = (category + np.random.randint(1,n_classes)) % n_classes
            pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w,h,1)
        return pairs, targets


class SiameseLoader(object):
    def __init__(self):
        self.training_data = {}
        self.testing_data = {}
        self.validation_data = {}
        self.img_size = (112, 92)

    def load_data(self,path,data_type='training'):
        data = self.training_data if data_type == 'training' else self.testing_data
        for person_id in os.listdir(path):
            data[person_id] = []
            print("Loading person ID: " + person_id)
            person_path = os.path.join(path,person_id)
            
            for person_image in os.listdir(person_path):
                image_path = os.path.join(person_path,person_image)
                image = imread(image_path)
                print("  -- Reading image " + person_image)
                data[person_id].append(image)
        return

    def random_pair(self,same=0.5,data_type='training'):
        imgs = []
        cls = ()
        data = self.training_data if data_type is 'training' else self.testing_data
        if np.random.random() > same:
            p = np.random.choice(list(data.keys()))
            rc = np.random.choice(len(data[p]),size=2,replace=False)
            imgs.append(data[p][rc[0]]), imgs.append(data[p][rc[1]])
            cls = (p, p)
        else:
            ps = np.random.choice(list(data.keys()),2,replace=False)
            imgs.append(random.choice(data[ps[0]]))
            imgs.append(random.choice(data[ps[1]]))
            cls = tuple(ps)
        return imgs, cls

    def get_training_batch(self,batch_size):
        '''Get Training Batch. Half of the instances are from the same class'''
        batch, labels = [], []
        for i in range(batch_size):
            if i > batch_size // 2:
                pair, cls = self.random_pair(-1) # same class
            else:
                pair, cls = self.random_pair(2) # different class
            batch.append(pair)
            labels.append(cls)
            
        pairs = [np.zeros((batch_size, *self.img_size, 1)) for _ in range(2)]
        targets = np.zeros((batch_size,))
        for i in range(batch_size):
            pairs[0][i,:,:,:] = batch[i][0].reshape(*self.img_size,1)
            pairs[1][i,:,:,:] = batch[i][1].reshape(*self.img_size,1)
        targets = [int(e1==e2) for e1,e2 in labels] # 1 for same class, 0 for different class
        return pairs, targets

    def make_oneshot_task(self,N,data_type='testing'):
        '''
        Return N one-shot pairs from the test set with only the first pair belonging
        to the same person
        Returns :
        	- test_images (numpy 4d array of size N x img_size x img_size x 1)
        	- support_set (numpy 4d array of size N x img_size x img_size x 1)
        '''
        data = self.testing_data if data_type is 'testing' else self.training_data
        # get random pair from same class
        same_pairs, true_class = self.random_pair(-1,data_type)
        # print('Testing Class : {}'.format(true_class))
        ptest1, ptest2 = same_pairs
        test_images = np.zeros((N, *self.img_size, 1))

        for i in range(N):            
            test_images[i,:,:,:] = ptest1.reshape(*self.img_size,1)
        
        support_set = np.zeros((N, *self.img_size, 1))
        support_classes = []
        support_set[0,:,:,:] = ptest2.reshape(*self.img_size,1)
        for i in range(1,N):
            random_face, cls = self.random_pair(2,data_type)
            while cls[0] == true_class[0]:
                random_face, cls = self.random_pair(2,data_type)
            support_set[i,:,:,:] = random_face[0].reshape(*self.img_size,1)
            support_classes.append(cls[0])

        return [test_images, support_set], true_class, support_classes

def load_training(path):
    person_dict = {}
    for person_id in os.listdir(path):
        person_dict[person_id] = []
        print("Loading person ID : " + person_id)
        person_path = os.path.join(path, person_id)

        for person_image in os.listdir(person_path):
            image_path = os.path.join(person_path,person_image)
            image = imread(image_path)
            person_dict[person_id].append(image)

    return person_dict

def random_pair(data,same=0.5):
    imgs = []
    cls = ()
    if np.random.random() > same:
        p = np.random.choice(list(data.keys()))
        rc = np.random.choice(len(data[p]),size=2,replace=False)
        imgs.append(data[p][rc[0]]), imgs.append(data[p][rc[1]])
        cls = (p, p)
    else:
        ps = np.random.choice(list(data.keys()),2,replace=False)
        imgs.append(random.choice(data[ps[0]]))
        imgs.append(random.choice(data[ps[1]]))
        cls = tuple(ps)
    return imgs, cls

def show_pair(p):
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(p[0])
    plt.subplot(2,2,2)
    plt.imshow(p[1])
    plt.show()
