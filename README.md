# Diabetes_Prediction
Diabetes_Prediction Using Support Vector Classifier

# Libraries Used :

1.Numpy
2.Pandas
3.Standarad Scalar
4.Train Test Split
5.Support Vector Machine
6.Accuracy Score

# Data Collection and Analysis

1.Load The Data
2.Then Used 'Shape' to get number of rows and Columns in this dataset
3.Then Getting The statistical measures of the data using 'describe'
4.Then Used 'Value Counts' on "OUTCOMES" to determine How Many Are Diabetic and How Many Are Not Diabetic
5.Then we are Taking The 'Mean' on "OUTCOMES" By using 'groupby'
6.Then separating the data and labels As Indepedent(X) and Dependent(Y) ### X Is data Y is Label

# Data Standardization

1.The values from the data are ranging between any values .so we need to standardize the data with Standard scalar. using fit It will fit the indepedent vectors and using transform it will transforms values between mean = 0 and standard deviation = 1.
2.Then we are giving stanadardized data to X . "OUTCOMES" variables assingned to Y

# Train Test Split

1.Here We will do Train Test Split . we will Mention 4 Variables X_train, X_test, Y_train, Y_test and 20% of data assigned for testing
2.Then We use 'Shape' For Knowing How Much data is been there in training and testing

# Training the Model

1.Here we are using support vector classifier . kernel = linear
2.Training the support vector Machine Classifier using fit And it is stored in classifier.

# Model Evaluation
 
## Accuracy Score

### Accuracy score on the training data

1. Here we are Using classifier to predict the lables (X_Train) and stored in X_train_prediction 
2. Then using X_train_prediction and Y_train . we can able to find training_data_accuracy

### Accuracy score on the training data

1. Here we are Using classifier to predict the lables (X_Test) and stored in X_test_prediction 
2. Then using X_test_prediction and Y_test . we can able to find testing_data_accuracy

# Making a Predictive System

we need to give input data[]
 
1.changing the input_data to numpy array
2.reshape the array as we are predicting for one instance
3.standardize the input data
4.if prediction[0] == 0
  The person is not diabetic
else
  The person is diabetic




