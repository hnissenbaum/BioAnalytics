### Author - Hannah Nissenbaum (Student #: 20049077)
import numpy as np
from sklearn import neighbors,model_selection,metrics
import matplotlib.pyplot as plt
import random



def feature_selection(train_set,test_set,train_labels,gene_num):
    train_set = train_set.transpose()
    test_set = test_set.transpose()
    # print(train_set.shape)

    features = len(train_set[:,1])
    data_pts = len(train_set[1,:])
    num1 = np.count_nonzero(train_labels)
    num0 = 30-num1
    #calculating the average of each feature through every datapoint
    features_avg0 = np.zeros((features))
    features_avg1 = np.zeros((features))
    data_sum0 = np.zeros((features))
    data_sum1 = np.zeros((features))
    for i in range(features): #through features
        for j in range(data_pts): #through points
            if train_labels[j] == 1:
                data_sum1[i] += train_set[i,j] #if corresponding to class 1, they'll be negative.
            else: #if half the feature points are classified as class1, half as class 2, the feature score will be close to zero (weak correlation)
                data_sum0[i] += train_set[i,j] #if half the feature points are classified as class1, half as class 2, the feature score will be close to zero
        features_avg0[i] = data_sum0[i]/num0
        features_avg1[i] = data_sum1[i]/num1


    #signal noise ratio - how far each feature value is from the average of the points in the data
    sig_noise_ratio0 = np.zeros((features))
    sig_noise_ratio1 = np.zeros((features))
    for i in range(features): #through features
        for j in range(data_pts): #through points
            if train_labels[j] == 1:
                sig_noise_ratio0[i] += (train_set[i,j] - features_avg0[i])**2
            else:
                sig_noise_ratio1[i] += (train_set[i,j] - features_avg1[i])**2
        sig_noise_ratio0[i] = np.sqrt(sig_noise_ratio0[i]/num0)
        sig_noise_ratio1[i] = np.sqrt(sig_noise_ratio1[i]/num1)

    #finding the features most indicative of each class through the signal noise ratio - signal(class0-class1)/noise(class0+class1) 
    #with this equation the scores that most strongly correspond to class 1 will be large and negative (small signal noise ratios - few outliers)
    feature_score = np.zeros((features))
    for i in range(features): #through features
            feature_score[i] = (features_avg0[i]-features_avg1[i])/(sig_noise_ratio0[i]+sig_noise_ratio1[i])

    #Finding the feature scores that are the largest/smallest (therefore demostrating strong correlations with class 1 or 2)
    sort_idx = np.argsort(feature_score) #best and worst scores in order
    selected_features = np.zeros((gene_num,data_pts))
    test_selected_features = np.zeros((gene_num,data_pts))
    selected_scores = np.zeros((features))
    distributed_nfeatures = np.zeros((features,data_pts))
    feat_idx = []
    for i,feat in enumerate(sort_idx[0:((gene_num))]): #top 500 genes corresponding to the non responders
        feat_idx.append(feat)
        selected_features[i,:] = (train_set[feat,:])
        test_selected_features[i,:] = (test_set[feat,:])
    # start = 7129-(int(gene_num/3))

    # for i,feat in enumerate(sort_idx[start:7129]): #top 500 genes corresponding to the non responders
    #     feat_idx.append(feat)
    #     selected_features[i+(int(gene_num/2)),:] = (train_set[feat,:])
    #     test_selected_features[i+(int(gene_num/2)),:] = (test_set[feat,:])
       
    import csv
    header = ['Data Point Index']
    with open('selected_features.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(feat_idx)

    return selected_features, test_selected_features