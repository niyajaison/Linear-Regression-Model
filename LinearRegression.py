'''
Name : Niya Jaison
The program does:
    1. Training a model via linear regression
    2. Using the trained model to do the classification
    3. Calcualte the best N value to choose for N-fold cross validation

'''
import pandas as data
import numpy

# Reading the data from the iris and adding the feature names

featureName = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'flowerClass']

irisData = data.read_csv('iris_data.csv',header = None,names = featureName)
#print(irisData)
foldValues=[3,5,10]

# Splitting data in to X and Y

X = irisData[['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']]
#X = irisData[[0,1,2,3]]
#print(X.values)
Y =irisData.flowerClass.map({'Iris-setosa':1.0,'Iris-versicolor':2.0,'Iris-virginica':3.0})
foldAccuracy=[0.0,0.0,0.0]
correct_label=0
p=0
fold_accuracy_per_fold=[0]
# Performing k-fold cross validation
for i in foldValues :
    del fold_accuracy_per_fold[0:len(fold_accuracy_per_fold)] 
    splitValue = len(X)/i #calculating the intervals based of the k-fold k value
    #print(splitValue)
    for j in range(i) :
        splt_lower=int(j*splitValue)
        splt_upper=int((j+1)*splitValue)
        # Choosing the 
        xTest=numpy.array(X[splt_lower:splt_upper])
        yTest=numpy.array(Y[splt_lower:splt_upper])
        xTrain =list(X.values)
        yTrain=list(Y.values)
        del xTrain[splt_lower : splt_upper : 1]
        del yTrain[splt_lower : splt_upper : 1]
       # del xTrain[2:3]
       # print(len(xTrain))
        xTrain= numpy.array(xTrain)
        yTrain= numpy.array(yTrain) 
        # beta = (X'X)^-1 X'Y - Creating a training Model
        beta = numpy.row_stack(numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(xTrain.T,xTrain)),xTrain.T),yTrain))

        #print(beta)

        # Calculating Y = X. beta - Performing Classification

        yTest_pred =numpy.row_stack( numpy.dot(xTest,beta))
        
        '''print(yTest_pred)
        #print(numpy.round(yTest_pred))
        #print(len(yTest))
        #print(len(yTest_pred))'''
        
        yTest_pred = numpy.round(yTest_pred) # Rounding of the predicted value to the nearest whole number( labels -1 or 2 or 3)
        for k in range(len(yTest_pred)) : # Checking the accuracy of the predicted y value with actual y value
            if(yTest_pred[k] == yTest[k]) :
                correct_label +=1
        fold_accuracy_per_fold.insert(j,correct_label/splitValue)
        correct_label=0
        #print(fold_accuracy_per_fold)
    foldAccuracy.insert(p,sum(fold_accuracy_per_fold)/(len(fold_accuracy_per_fold)))
    p+=1
    #print(foldAccuracy)
    del fold_accuracy_per_fold[0:len(fold_accuracy_per_fold)] # clearing the list to store the accuarcy of each of the fold
    #print(fold_accuracy_per_fold)    
    
highest_accuracy_fold = foldAccuracy.index(max(foldAccuracy)) # finding the index of the fold accuracy which has max value
#print(foldAccuracy)

print('Selected N value (in N-fold) for cross validation :',foldValues[highest_accuracy_fold],'(Accuracy : ',foldAccuracy[2]*100,'%)\n')

for l in range(foldValues[highest_accuracy_fold]) :
        splt_lower=int(l*splitValue)
        splt_upper=int((l+1)*splitValue)
        # Choosing the 
        xTest=numpy.array(X[splt_lower:splt_upper])
        yTest=numpy.array(Y[splt_lower:splt_upper])
        xTrain =list(X.values)
        yTrain=list(Y.values)
        del xTrain[splt_lower : splt_upper : 1]
        del yTrain[splt_lower : splt_upper : 1]
       # del xTrain[2:3]
       # print(len(xTrain))
        xTrain= numpy.array(xTrain)
        yTrain= numpy.array(yTrain) 
        # beta = (X'X)^-1 X'Y - Creating a training Model
        beta = numpy.row_stack(numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(xTrain.T,xTrain)),xTrain.T),yTrain))
        print('Values calculated when considering section ',l+1,'  as test data:\n')
        print('Beta for the test data:\n',beta)

        # Calculating Y = X. beta - Performing Classification

        yTest_pred =numpy.row_stack( numpy.dot(xTest,beta))
        
        '''print(yTest_pred)
        #print(numpy.round(yTest_pred))
        #print(len(yTest))
        #print(len(yTest_pred))'''
        
        yTest_pred = numpy.round(yTest_pred)
        print('Test Value :\n',xTest)
        print('Predicted Value:\n',yTest_pred)
      #  print(xTrain)
       

