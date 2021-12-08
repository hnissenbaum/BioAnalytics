### Author - Hannah Nissenbaum (Student #: 20049077)
import numpy as np
from sklearn import neighbors,model_selection,metrics
import matplotlib.pyplot as plt
import random
from main_knn import data_randomize

#importing the data
file = open("dataclass1.csv"); dataclass1 = np.loadtxt(file, delimiter=",")
file = open("dataclass2.csv"); dataclass2 = np.loadtxt(file, delimiter=",")

train_set,train_labels,test_set,test_labels = data_randomize(dataclass1,dataclass2)
train_set = train_set.transpose()
test_set = test_set.transpose()

features = len(train_set[:,1])
data_pts = len(train_set[1,:])

#calculating the average of each feature through every datapoint
features_avg = np.zeros((features))
data_sum = np.zeros((features))
count = 0
for i in range(features): #through features
    for j in range(data_pts): #through points
        data_sum[i] += train_set[i,j]
    features_avg[i] = data_sum[i]/data_pts


#signal noise ratio - how far each feature value is from the average of the points in the data
sig_noise_ratio = np.zeros((features,data_pts))
for i in range(features): #through features
    for j in range(data_pts): #through points
        sig_noise_ratio[i,j] = abs(train_set[i,j] - features_avg[i])
        if sig_noise_ratio[i,j] == 0:
            sig_noise_ratio[i,j] = np.inf #if the data is zero - it's not a feature we want to select

#finding the features most indicative of each class through the signal noise ratio - signal(class0-class1)/noise(class0+class1) 
#with this equation the scores that most strongly correspond to class 1 will be large and negative (small signal noise ratios - few outliers)
feature_score = np.zeros((features))
for i in range(features): #through features
    temp =0
    for j in range(data_pts): #through points
        if train_labels[j] == 1:
            temp += (-1)*(features_avg[i])/sig_noise_ratio[i,j] #if corresponding to class 1, they'll be negative.
        else: #if half the feature points are classified as class1, half as class 2, the feature score will be close to zero (weak correlation)
            temp += features_avg[i]/sig_noise_ratio[i,j] #if half the feature points are classified as class1, half as class 2, the feature score will be close to zero
    feature_score[i] = temp

#Finding the feature scores that are the largest/smallest (therefore demostrating strong correlations with class 1 or 2)
sort_idx = np.argsort(feature_score)
selected_features = np.zeros((1000,data_pts))
selected_scores = np.zeros((features))
for i in range(500):
    selected_features[i] = (train_set[sort_idx[i],:])
    selected_features[999-i] = (train_set[(sort_idx[(features-1)-i])-i,:])
    selected_scores[sort_idx[i]] = feature_score[sort_idx[i]]
    selected_scores[sort_idx[(features-1)-i]] = feature_score[sort_idx[(features-1)-i]]


plt.figure(1)
sig = np.zeros((features))
for i in range(features):
        sig[i] = np.average(sig_noise_ratio[i,:])
plt.bar(range(features),sig)
plt.title("Features - Signal to Noise Ratio")
plt.xlabel("Feature Index")
plt.ylabel("Occurances of Difference Value")
plt.ylim((0,4000))
plt.xlim((0,7129))


plt.figure(2)
plt.bar(range(features),np.sort(feature_score))
plt.bar(range(features),np.sort(selected_scores))
plt.title("Feature Score")
plt.xlabel("Feature Index")
plt.ylabel("Feature Corralation") #(if <0, corralated to Class 1, if >0, corralated to Class 2)
plt.legend(['Feature Scores','Selected Feature Scores'])
plt.ylim((-10000,10000))
plt.show()






