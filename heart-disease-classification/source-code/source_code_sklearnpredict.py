# import the packages that we'd work with.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Load the data into a pandas dataframe.
df = pd.read_csv(r"c:\Users\USER\Desktop\MyDatasets\heart.csv")
df.head()

# Replace the categorical attributes with numbers.
df.Sex.replace(['F', 'M'], [0, 1], inplace=True)
df.ChestPainType.replace(['ATA', 'NAP', 'ASY', 'TA'], [0, 1, 2, 3], inplace=True)
df.RestingECG.replace(['Normal', 'ST', 'LVH'], [0, 1, 2], inplace=True)
df.ExerciseAngina.replace(['N', 'Y'], [0, 1], inplace=True)
df.ST_Slope.replace(['Up', 'Flat', 'Down'], [0, 1, 2], inplace=True)
df.head()

# Convert the dataframe to a numpy array.
data = df.to_numpy()

# Slice the data into features and targets.
X = data[:, :11]
Y = data[:, 11]

# Split the features and targets into training and testing chunks
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=1, shuffle=True)

# Scale or normalised the input data(features)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=1)
dtc.fit(X_train_scaled, Y_train)
accuracy = round(((dtc.score(X_test_scaled, Y_test)) * 100 ))
# print(f'Accuracy of the Decision Tree Classifier: {accuracy}%')  --uncomment to print--

# K-Fold with Decision Tree
dtc = DecisionTreeClassifier(random_state=1)
cv_scores = cross_val_score(dtc, X, Y, cv=26)
accuracy = round((cv_scores.mean()) * 100)
# print(f'Accuracy: {accuracy}%')

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=35, random_state=1)
rfc.fit(X_train_scaled, Y_train)
accuracy = round((rfc.score(X_test_scaled, Y_test) * 100))
# print(f'Accuracy of Random Forest Classifier: {accuracy}%')

# SVM
svc = svm.SVC(kernel='rbf', C=1.0)
svc.fit(X_train_scaled, Y_train)
accuracy = round((svc.score(X_test_scaled, Y_test))*100)
# print(f'Accuracy of the SVM: {accuracy}%')

# KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_scaled, Y_train)
accuracy = round((knn.score(X_test_scaled, Y_test)) * 100)
# print(f'Accuracy of model: {accuracy}%')

# Testing K value loop, unquote to run the below snippet
'''for i in range(1, 51):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, Y_train)
    accuracy = round((knn.score(X_test_scaled, Y_test))*100)
    print(f'Accuracy of model using K value of {i}: {accuracy}%')'''
    
# Naive Bayes
# scale the input features with minmax scaler
scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(X_train)
X_test_minmax = scaler.transform(X_test)

# define the classifier
nb = MultinomialNB()
nb.fit(X_train_minmax, Y_train)
accuracy = round((nb.score(X_test_minmax, Y_test)) * 100)
# print(f'Accuracy of the model: {accuracy}%')

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, Y_train)
accuracy = round((lr.score(X_test_scaled, Y_test))*100)
# print(f'Accuracy of the model: {accuracy}%')



# Re-testing all the models

# I standardised the data prior to.
patient_data = np.array([[-0.5789,  0.5220,  0.6349, -0.9879, -1.7750,  1.8279, -0.7600, -0.3676,
         -0.8088, -0.8119,  0.6583], [-1.3316,  0.5216,  0.6352, -1.1563,  0.4743, -0.5473, -0.7599,  0.1890,
         -0.8082, -0.8111, -1.0494]])


'''These loops have to be run individually after fitting the respective 
classifier to the actual training data, before calling predict, so that it doesn't return an error
'''                                      
print(f'**Decision Tree Classifier**')
for i in range(len(patient_data)):
    data = patient_data[i].reshape(1, -1)
    predicted = dtc.predict(data)
    if predicted==1:
        print(f'Patient {i+1} has a heart disease')
    else:
        print(f'Patient {i+1} does not have a heart disease')
        
print(f'\n**Random Forest Classifier**')
for i in range(len(patient_data)):
    data = patient_data[i].reshape(1, -1)
    predicted = rfc.predict(data)
    if predicted==1:
        print(f'Patient {i+1} has a heart disease')
    else:
        print(f'Patient {i+1} does not have a heart disease')
        
print(f'\n**SVM**')
for i in range(len(patient_data)):
    data = patient_data[i].reshape(1, -1)
    predicted = svc.predict(data)
    if predicted==1:
        print(f'Patient {i+1} has a heart disease')
    else:
        print(f'Patient {i+1} does not have a heart disease')
        
print(f'\n**KNN**')
for i in range(len(patient_data)):
    data = patient_data[i].reshape(1, -1)
    predicted = knn.predict(data)
    if predicted==1:
        print(f'Patient {i+1} has a heart disease')
    else:
        print(f'Patient {i+1} does not have a heart disease')
        
print(f'\n**Naive Bayes**')
for i in range(len(patient_data)):
    data = patient_data[i].reshape(1, -1)
    predicted = nb.predict(data)
    if predicted==1:
        print(f'Patient {i+1} has a heart disease')
    else:
        print(f'Patient {i+1} does not have a heart disease')
        
print(f'\n**Logistic Regression**')
for i in range(len(patient_data)):
    data = patient_data[i].reshape(1, -1)
    predicted = lr.predict(data)
    if predicted==1:
        print(f'Patient {i+1} has a heart disease')
    else:
        print(f'Patient {i+1} does not have a heart disease')
        
# ifunanyaScript