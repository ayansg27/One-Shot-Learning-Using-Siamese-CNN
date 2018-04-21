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

    def random_pair(self,same=0.5):
        imgs = []
        cls = ()
        data = self.training_data
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

    def batch(self,batch_size,data_type='training'):
        if data_type is 'training':
            data = self.training_data
        else:
            data = self.testing_data

        batch, labels = [], []
        for i in range(batch_size):
            if i > batch_size//2:
                pair, cls = self.random_pair(2)
            else:
                pair, cls = self.random_pair(-1)
            batch.append(pair)
            labels.append(cls)
        return batch, labels

    def get_batch(self,batch_size):
        data = self.training_data
        batch, labels = [], []
        for i in range(batch_size):
            pair = self.random_pair()
            batch.append(pair[0])
            labels.append(pair[1])
        return batch, labels

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
