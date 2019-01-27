import os
import numpy as np
import matplotlib.pyplot as plt

class Sample:
    def __init__(self):
        self.img = []
        self.pose = []

    def initial(self, img, pose):
        self.img = img
        self.pose = pose


def getSubDir(path):
    return next(os.walk(path))[1]

def assertDimsCheck(set):
    print(set["img"].shape, set["pose"].shape,set["label"].shape)


def read_indicies(file_path,file_name):

    num_class = 5
    img_per_class = 1178
    train_idx = []
    f = open(os.path.join(file_path, file_name), 'r')
    line = f.readline()
    indices = line.split(',')
    indices = list(map(int, indices))
    f.close()
    # print(indices)
    # print(int(indices[-1]))
    for i in range(num_class):
        for j in indices:
            #print(i*img_per_class + j)
            train_idx.append(i*img_per_class + j)

    test_idx = [i for i in range(img_per_class * num_class) if i not in train_idx]

    return train_idx,test_idx

def loadDataset(dataSetPath,poseFilePath):

    # define the data container
    quateronins_list = []
    img_list = []
    class_list = []
    dataset = {}

    classes = getSubDir(dataSetPath)
    print (classes)
    # print(classes[0])
    p = classes[3]

    # for filename in os.listdir(curr_dir):
    # if filename.endswith(".png"):
    # img_file = (os.path.join(curr_dir, filename))
    # print(img_file)
    # img  = plt.imread(img_file)
    # continue

    #if(p):
    for p in sorted(classes):
        #print(p)
        curr_dir = os.path.join(dataSetPath,p)
        pose_file = open(os.path.join(curr_dir,poseFilePath),'r')

        line = pose_file.readline()
        while line:
            if line.strip('\n').endswith(".png"):
                #print(line)
                img_file_name = line.split(" ")[1].strip('\n')
                #print(line.split(" ")[1].strip('\n'))
                img = plt.imread(os.path.join(curr_dir, img_file_name))
                img_list.append(img)
                #print(img.shape)

                line = pose_file.readline()
                qt_pose = line.split(" ")
                qt_pose[3] = qt_pose[3].strip('\n')
                quateronins_list.append(qt_pose)

                class_list.append(p)
                # print (qt_pose[3].strip('\n'))
            line = pose_file.readline()

        # bind the data into dicts and return the dict
        dataset['img'] = np.asarray(img_list)
        dataset['pose'] = np.asarray(quateronins_list)
        dataset['label'] = np.asarray(class_list)

    assertDimsCheck(dataset)
    return dataset

def build_train_set(real_set,fine_set,train_idx):
    print('building train set')
    #train_idx,test_idx = util.read_indicies(os.path.join(base_path, "dataset\\real"), 'training_split.txt')

    #definig containers
    train_dict = {}

    #adding real image
    img_train_dict = real_set['img'][train_idx[:], :, :, :]
    pose_train_dict = real_set['pose'][train_idx[:], :]
    label_train_dict = real_set['label'][train_idx[:]]

    # print(img_train_dict.shape)
    # print(pose_train_dict.shape)
    # print(label_train_dict.shape)
    #util.assertDimsCheck(train_set_dict)

    # adding the synthetic image
    # adding fine set to the test set
    processed_train = np.concatenate((img_train_dict,fine_set['img']), axis= 0)
    processed_train = (processed_train - np.mean(processed_train, axis = 0))
    processed_train = processed_train / np.std(processed_train,axis = 0)
    # print(processed_train.shape)
    train_dict['img'] = processed_train
    train_dict['pose'] = np.concatenate((pose_train_dict,fine_set['pose']), axis= 0)
    train_dict['label'] = np.concatenate((label_train_dict,fine_set['label']), axis= 0)

    assertDimsCheck(train_dict)
    return train_dict


def build_test_set(real_set,test_idx):
    print('building test set')
    #train_idx, test_idx = util.read_indicies(os.path.join(base_path, "dataset\\real"), 'training_split.txt')

    test_dict = {}
    test_list = real_set['img'][test_idx[:], :, :, :]
    processed_test = (test_list - np.mean(test_list, axis=0))
    processed_test = processed_test / np.std(processed_test, axis=0)
    #print(processed_test.shape)

    test_dict['img'] = processed_test
    test_dict['pose'] = real_set['pose'][test_idx[:], :]
    test_dict['label'] = real_set['label'][test_idx[:]]

    assertDimsCheck(test_dict)
    return test_dict

