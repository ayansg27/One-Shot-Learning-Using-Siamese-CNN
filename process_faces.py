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

class DataLoader(object):
    def __init__(self,path,data_subsets=['training','evaluation']):
        self.data = {}
        self.categories = {}
        self.info = {}

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
