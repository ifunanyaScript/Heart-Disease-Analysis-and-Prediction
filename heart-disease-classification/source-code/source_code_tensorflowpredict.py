# import packages that we'd work with
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import SGD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

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

# Convert the dataframe into a numpy array, so we can train on it.
data = df.to_numpy()

# Slice the data into features and targets.
X = data[:, :11]
Y = data[:, 11]

# Split the data into training and testing chunks.
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=2)

# Scale/normalize the data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Define the input_shape.
input_shape = (X_train_scaled.shape[1],)

# Build the model.
model = Sequential()
model.add(Dense(41, activation='relu', input_shape=input_shape))
#model.add(Dropout(0.2))
model.add(Dense(112, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile the model and define the loss and optimizer functions.
model.compile(loss='binary_crossentropy', 
                         optimizer=SGD(learning_rate=0.1), 
                         metrics=['accuracy'])

# Call model.fit on the training chunk, define batch size, epochs and  the validation data.
model.fit(X_train_scaled, Y_train,
                               batch_size=40, 
                               epochs=5,
                               verbose=2,
                               validation_data=(X_test_scaled, Y_test))
# Training!!!

# I've made up two samples of data, the first is a heart disease positive patient,
# the second is a heart disease negative patient. Let's see if our model can tell.
# I standardised the data prior to.
patient_data = np.array([[-0.5789,  0.5220,  0.6349, -0.9879, -1.7750,  1.8279, -0.7600, -0.3676,
         -0.8088, -0.8119,  0.6583], [-1.3316,  0.5216,  0.6352, -1.1563,  0.4743, -0.5473, -0.7599,  0.1890,
         -0.8082, -0.8111, -1.0494]])
                                      
predicted = model.predict(tf.convert_to_tensor(patient_data)) # you mustn't convert this to a tensor, it's not necessary.
for i in range(len(predicted)):
    prediction = (predicted[i]).round()
    if prediction==1:
        print(f'Patient {i+1} has a heart disease')
    else:
        print(f'Patient {i+1} does not have a heart disease')
        
# ifunanyaScript