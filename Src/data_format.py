import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def data_format(class1,class2): 
    r1,__ = class1.shape
    
    r2,__ = class2.shape
    #creating empty arrays for the trainingdataing data and the testingdata data
    #the data will be stored in a 3D matrix, with the third dimension containing the class
    trainingdata = np.zeros((max(r1,r2),30),dtype='int')
    testingdata = np.zeros((max(r1,r2),30),dtype='int')
    train_labels = np.zeros((30))
    test_labels = np.zeros((30))

    class1_idx = random.sample(range(0, 21), 21)
    class1_idx = np.asarray(class1_idx,dtype='int')
    class2_idx = random.sample(range(0, 39), 39) 
    class2_idx = np.asarray(class2_idx,dtype='int')
    
    for i in range(0,10): #additionally, all rows must be stored (together with their relevant data points)
        train_labels[i] = 0 #the labels for these data points are class 0 (bad outcomes)
        test_labels[i] = 0
        trainingdata[:,i] = class1[:,class1_idx[i]]
        #print(f"train idx: {class1_idx[i]},test idx: {class1_idx[20-i]} - with idx {i,20-i}")
        testingdata[:,i] = class1[:,class1_idx[20 - i]]
    
    #assigning the bias of the 21st datapoint in class 1 - randomly choosing an
    #integer to choose between the two datasets.
    bias = np.random.randint(0,2)
    if bias == 0:
        trainingdata[:,11] = class1[:,class1_idx[20]]
        train_labels[11] = 0
        trainingdata_indexing = 11
        testingdata_indexing = 10
        testingdata[:,29] = class2[:,class2_idx[38]]
        test_labels[29] = 1
        #print(f"now adding idx (11,21)")
    else:
        testingdata[:,11] = class1[:,class1_idx[20]]
        test_labels[11] = 0
        trainingdata_indexing = 10
        testingdata_indexing = 11
        trainingdata[:,29] = class2[:,class2_idx[38]]
        train_labels[29] = 1
        #print(f"now adding idx (11,21)")
    
    for i in range(0,19):
        trainingdata[:,trainingdata_indexing+i] = class2[:,class2_idx[i]]
        testingdata[:,testingdata_indexing+i] = class2[:,class2_idx[38 - i]]
        train_labels[trainingdata_indexing+i] = 1
        test_labels[testingdata_indexing+i] = 1
        #print(f"train idx: {class2_idx[i]},test idx: {class2_idx[38-1]} - with idx {trainingdata_indexing+i,testingdata_indexing+i}")
    
    # val_set = np.zeros((max(r1,r2),15),dtype='int')
    # test_set = np.zeros((max(r1,r2),15),dtype='int')
    # val_labels = np.zeros((15))
    # testing_labels = np.zeros((15))

    # val_test_idx = random.sample(range(0, 30), 30)
    # val_test_idx = np.asarray(val_test_idx,dtype='int')

    # for i in range(0,14):
    #     val_set[:,i]   = testingdata[:,val_test_idx[i]]
    #     test_set[:,i]  = testingdata[:,val_test_idx[29 - i]]
    #     val_labels[i]  = test_labels[val_test_idx[i]]
    #     testing_labels[i] = test_labels[val_test_idx[29-i]]
        # print(f"val idx: {val_test_idx[i]},test idx: {val_test_idx[29-i]} - with idx {i,29-i}")
    
    # plt.subplot(2,2,1)
    # plt.title('val')
    # plt.plot(range(1,16),val_set[0,:])

    # plt.subplot(2,2,2)
    # plt.title('test')
    # plt.plot(range(1,16),test_set[0,:])

    # plt.subplot(2,2,3)
    # plt.title('val idx')
    # plt.plot(range(1,16),val_labels)

    # plt.subplot(2,2,4)
    # plt.title('test idx')
    # plt.plot(range(1,16),testing_labels)
    # plt.show()
    # print(val_set.shape,test_set.shape,val_labels.shape,testing_labels.shape)

    print('Data loaded!')
    print(f"bias:{bias}")
    trainingdata = trainingdata.transpose()
# self.train_set = self.train_set.transpose()
    testingdata = testingdata.transpose()

    return trainingdata,train_labels,testingdata,test_labels #val_set,val_labels,test_set,testing_labels

# def plots():
    # plt.subplot(3,2,1)
    # plt.title('train data')
    # plt.plot(range(1,31),tr_data[0,:,0])

    # plt.subplot(3,2,3)
    # plt.title('train data')
    # plt.plot(range(1,31),tr_data[0,:,1])

    # plt.subplot(3,2,2)
    # plt.title('test data')
    # plt.plot(range(1,31),(tr_data[0,:,0]))

    # plt.subplot(3,2,4)
    # plt.title('test data')
    # plt.plot(range(1,31),test_data[0,:,1])

    # plt.subplot(3,2,5)
    # plt.title('train labels')
    # plt.plot(range(1,31),train_labels)

    # plt.subplot(3,2,6)
    # plt.title('test labels')
    # plt.plot(range(1,31),test_labels)



    # print(train_labels, test_labels)
    # # print(tr_data)
    # # print(test_data)
    # #plt.show()