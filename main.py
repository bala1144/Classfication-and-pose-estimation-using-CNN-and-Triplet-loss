import os
import numpy as np
import matplotlib.pyplot as plt
import util as util
import cv2




print('Hello world')
#print(sys.executable)

#steps for data generation

#load the image from the path
#function to read corresponding pose from the file (pose are stored in quaterions)
#create a class or struct for the samples

#s_train =  real + fine
#s_db = coarse folder
#s_test = real

#batch generator function -> triplet generation
# anchor = randomly chosen from the training set
# Puller = most similar to anchor from the db
# pusher = same object different pose and different object pusher

def visualizer(anchor,puller,s_pusher):
    cv2.imshow("anchor ", anchor)
    cv2.imshow("puller", puller)
    cv2.imshow("same class pusher", s_pusher)
    cv2.waitKey(0)

def get_puller(anchor,anc_class,s_db):
    #print('building puller')
    #pusher_thershold = 10
    thetha_list = []
    puller = -1

    #select only sample from the class
    indx = [i for i in range(s_db['pose'].shape[0]) if s_db['label'][i] ==  anc_class]
    #print('indx', indx)

    for idx in indx:
        test_pusher = s_db['pose'][idx, :]
        anchor = list(map(float, anchor))
        test_pusher = list(map(float, test_pusher))

        x = np.absolute(np.dot(test_pusher, anchor))
        theth_var = 2 * np.arccos(x)
        thetha_list.append(theth_var)
        #print(theth_var)

    puller = np.argmin(thetha_list)

    #same class pusher
    s_pusher = np.argmax(thetha_list)
    #need to change it

    # return puller + indx[0], puller + indx[0] + pusher_thershold
    return puller + indx[0], s_pusher + indx[0]


def gen_triplet_list(s_train,s_db):
    print('building triplet list')
    triplet_list = []

    #for i in range(10):
    for i in range(s_train['pose'].shape[0]):
        #print(i)
        test_anchor = s_train['pose'][i, :]
        test_anchor_class = s_train['label'][i]
        # print('anchor class', s_train['label'][i])

        puller, s_pusher = get_puller(test_anchor, test_anchor_class, s_db)
        # print('puller class', s_db['label'][puller])
        triplet_list.append([i,puller,s_pusher])
    # visualizer(s_train['img'][i, :, :, :], s_db['img'][puller, :, :, :], s_db['img'][s_pusher, :, :, :])

    print('number of triplets = ', len(triplet_list))
    #return triplet_list

def get_batch(triplet_list,batch_size):
    batch = {}
    





if '__main__' == __name__:

    base_path = os.getcwd()
    print('path = ', base_path)

    fine_set = util.loadDataset(os.path.join(base_path, "dataset\\fine"), 'poses.txt')
    s_db = util.loadDataset(os.path.join(base_path, "dataset\\coarse"), 'poses.txt')
    real_set = util.loadDataset(os.path.join(base_path, "dataset\\real"), 'poses.txt')
    train_idx, test_idx = util.read_indicies(os.path.join(base_path, "dataset\\real"), 'training_split.txt')
    s_train = util.build_train_set(real_set,fine_set,train_idx)
    s_test = util.build_test_set(real_set,test_idx)

    gen_triplet_list(s_train,s_db)
