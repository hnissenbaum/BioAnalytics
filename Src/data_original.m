function [trainingdata, testingdata,bias,trainingdat,testingdat,training_label,testing_label] = data_original(class1,class2) %#ok<*FNDEF>
[r1,~] = size(class1); %rows, columns
[r2,~] = size(class2);

%creating empty arrays for the trainingdataing data and the testingdata data
%the data will be stored in a 3D matrix, with the third dimension containing the class
trainingdata = zeros(max(r1,r2),20,2);
testingdata = zeros(max(r1,r2),20,2);
training_label = blanks(30);
testing_label = blanks(30);
trainingdat = zeros(7129,30);
testingdat = zeros(7129,30);


class1_idx = randperm(21,21);
class2_idx = randperm(39,39);
for i = 1:10
    %storing this data in 'sheet 1' or the first index of the third
    %dimension.
    %additionally, all rows must be stored (together with their relevant
    %data points)
    trainingdata(:,i,1) = class1(:,class1_idx(i)); %use the random string of integers as an index in class one
    testingdata(:,i,1) = class1(:,class1_idx(21-i)); %because it's a string of unique integers, the other half of the random string is used for the testingdata data
    training_label(i) = 'b';
    testing_label(i) = 'b';
    trainingdat(:,i) = class1(:,class1_idx(i));
    testingdat(:,i) = class1(:,class1_idx(21-i));

end
%assigning the bias of the 21st datapoint in class 1 - randomly choosing an
%integer to choose between the two datasets.
bias = randi(2);
if bias==1
    trainingdata(:,11,1) = class1(:,class1_idx(21));
    trainingdata_indexing = 0;
    testingdata_indexing = 1;
    training_label(11) = 'b';
    testing_label(11) = 'g';
    trainingdat(:,11) = class1(:,class1_idx(21));
    testingdat(:,11) = class2(:,class2_idx(39));
    testingdata(:,1,2) = class2(:,class2_idx(39)); %assigning the last random integer indexed column to class 2 testingdata data
else 
    testingdata(:,11,1) = class1(:,class1_idx(21));
    trainingdata_indexing = 1;
    testingdata_indexing = 0;
    training_label(11) = 'g';
    testing_label(11) = 'b';
    testingdat(:,11) = class1(:,class1_idx(21));
    trainingdat(:,11) = class2(:,class2_idx(39));
    trainingdata(:,1,2) = class2(:,class2_idx(39));
    
end

for i = 1:19
    trainingdata(:,trainingdata_indexing+i,2) = class2(:,class2_idx(i));
    testingdata(:,testingdata_indexing+i,2) = class2(:,class2_idx(39-i));
    training_label(i+11) = 'g';
    testing_label(i+11) = 'g';
    trainingdat(:,i+11) = class2(:,class2_idx(i));
    testingdat(:,i+11) = class2(:,class2_idx(39-i));
end
fprintf("Data loaded!");
return