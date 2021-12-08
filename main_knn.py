### Author - Hannah Nissenbaum (Student #: 20049077)
import numpy as np
from sklearn import neighbors,model_selection,metrics
import matplotlib.pyplot as plt
import random
score_metric = metrics.make_scorer(metrics.accuracy_score)



def data_randomize(class1,class2): 
    r1,__ = class1.shape
    r2,__ = class2.shape

    #creating empty arrays for the training/testing data
    #the data will be stored in a 3D matrix, with the third dimension containing the class
    trainingdata = np.zeros((max(r1,r2),30),dtype='int')
    testingdata = np.zeros((max(r1,r2),30),dtype='int')
    train_labels = np.zeros((30))
    test_labels = np.zeros((30))

    #generating randomized strings of integers for indexing each dataset
    class1_idx = random.sample(range(0, 21), 21)
    class1_idx = np.asarray(class1_idx,dtype='int')
    class2_idx = random.sample(range(0, 39), 39) 
    class2_idx = np.asarray(class2_idx,dtype='int')
    
    for i in range(0,10): #additionally, all rows must be stored (together with their relevant data points)
        train_labels[i] = 0 #the labels for these data points are class 0 (bad outcomes)
        test_labels[i] = 0
        trainingdata[:,i] = class1[:,class1_idx[i]]
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
    else:
        testingdata[:,11] = class1[:,class1_idx[20]]
        test_labels[11] = 0
        trainingdata_indexing = 10
        testingdata_indexing = 11
        trainingdata[:,29] = class2[:,class2_idx[38]]
        train_labels[29] = 1
    
    for i in range(0,19):
        trainingdata[:,trainingdata_indexing+i] = class2[:,class2_idx[i]]
        testingdata[:,testingdata_indexing+i] = class2[:,class2_idx[38 - i]]
        train_labels[trainingdata_indexing+i] = 1
        test_labels[testingdata_indexing+i] = 1
    
    #transpose for KNN functions
    trainingdata = trainingdata.transpose()
    testingdata = testingdata.transpose()

    return trainingdata,train_labels,testingdata,test_labels




#importing the data
file = open("tr.csv"); train_set = np.loadtxt(file, delimiter=",")
file2 = open("te.csv"); test_set = np.loadtxt(file2, delimiter=",")

#collecting data randomly seperated by matlab
train_labels = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
test_labels = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


#train_set,train_labels,test_set,test_labels = data_randomize(dataclass1,dataclass2)

#defining the weighted voting scheme for cross validation
def weighting(scores,k_val):
    #the more accurate k values get a larger weight in the vote then the less accurate ones
    kval = np.average(scores) #find the average accuracy score for this value of k
    return kval,k_val

ks = range(2,10)
k_score = []
for k in ks:
    # train the model with the whole training set and the current k
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_set, train_labels)
    
    #setting up cross validation
    loo = model_selection.LeaveOneOut()
    cv = model_selection.cross_val_score(knn, train_set, train_labels, scoring=score_metric, cv=model_selection.KFold(n_splits=25))
    
    #finding the best accuracy found in cross validation
    scoring = cv
    score,k = weighting(scoring,k)
    k_score.append(score)

# Fitting Cross Validation Results and Evaluating
ind = np.argmax(k_score) #the maximum average accuracy score is the k winner - determined by the indices of the table.
best_k = ks[ind]
print(f"Best K Determined to be: {best_k}")

# re-fit the model with best-k
knn = neighbors.KNeighborsClassifier(n_neighbors=best_k)
knn.fit(train_set, train_labels)
results = knn.predict(test_set)
accuracy_test = metrics.accuracy_score(test_labels, results)
c1 = np.zeros((30,3), dtype='int')
c2 = np.zeros((30,1), dtype='int')
c3 = np.zeros((30,1), dtype='int')
import csv  


for i in range(1,31):
    print(f"Data Point Index: {i}     |       Predicted Class: {results[i-1]}       |       True Class: {test_labels[i-1]}")

    c1[i-1] = [i, results[i-1], test_labels[i-1]]

header = ['Data Point Index', 'Predicted Class', 'True Class']
with open('finalresults.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(c1)



print("Total Accuracy on Test Set: %2.1f %%" %(accuracy_test*100))