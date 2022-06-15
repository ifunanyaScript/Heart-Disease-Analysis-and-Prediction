# import the packages we'd work with
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

# Load data into pandas dataframe.
df = pd.read_csv(r"c:\Users\USER\Desktop\MyDatasets\heart.csv")
df.head(5)

# We have to replace the categorical attributes with numbers.
df.Sex.replace(['F', 'M'], [0, 1], inplace=True)
df.ChestPainType.replace(['ATA', 'NAP', 'ASY', 'TA'], [0, 1, 2, 3], inplace=True)
df.RestingECG.replace(['Normal', 'ST', 'LVH'], [0, 1, 2], inplace=True)
df.ExerciseAngina.replace(['N', 'Y'], [0, 1], inplace=True)
df.ST_Slope.replace(['Up', 'Flat', 'Down'], [0, 1, 2], inplace=True)
df.head(5)

# We then convert the dataframe into a numpy array so we can train on it.
data = df.to_numpy()

# We slice data into features and targets.
X = data[:, :11]
Y = data[:, [11]]

# We split the data into training and testing chunks.
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=1, shuffle=True)

# We scale/normalize the data.
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# We convert the arrays to tensors.
X_train_tensor = torch.from_numpy(X_train_scaled.astype(np.float32))
Y_train_tensor = torch.from_numpy(Y_train.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32))
Y_test_tensor = torch.from_numpy(Y_test.astype(np.float32))

# We set the number of features that the neural network expects.
input_features = X_train_tensor.shape[1]

# Build the model and call sigmoid at the end.
class Model(nn.Module):
    def __init__(self, input_features):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_features, 10000)
        self.l_relu = nn.LeakyReLU()
        self.l2 = nn.Linear(10000, 2)
        self.l3 = nn.Linear(2, 1)
    def forward(self, x):
        out = self.l1(x)
        out = self.l_relu(out)
        out = self.l2(out)
        out = self.l_relu(out)
        out = self.l3(out)
        out = torch.sigmoid(out)
        return out
    
# We initialize our model, and we define the loss and optimizer functions.
model = Model(input_features)
criterion = nn.BCELoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop!
epochs = 35
for epoch in range(epochs):
    # forward pass and computing loss!
    pred = model(X_train_tensor)
    loss = criterion(pred, Y_train_tensor)
    
    # empty gradients!
    optimizer.zero_grad()
    # calculate gradients and backward pass!
    loss.backward()
    # update model parameters(weights and biases)!
    optimizer.step()
    
    # Training info!
    if (epoch%5)==0:
        print(f'Epoch: {epoch}/{epochs}, Loss: {loss:.10f}')
        
# Model evaluation!
with torch.no_grad():
    predicted = model(X_test_tensor)
    prediction = predicted.round()
    accuracy = (prediction.eq(Y_test_tensor).sum()/Y_test_tensor.shape[0])*100
    print(f'Accuracy: {accuracy.round()}%')
    

# I've made up two samples of data, the first is a heart disease positive patient, 
#the second is a heart disease negative patient. Let's see if our model can tell.
# I standardised the data prior to.
patient_data = np.array([[-0.5789,  0.5220,  0.6349, -0.9879, -1.7750,  1.8279, -0.7600, -0.3676,
         -0.8088, -0.8119,  0.6583], [-1.3316,  0.5216,  0.6352, -1.1563,  0.4743, -0.5473, -0.7599,  0.1890,
         -0.8082, -0.8111, -1.0494]])
# convert to tensor
patient_data_tensor = torch.from_numpy(patient_data.astype(np.float32))

with torch.no_grad():
    for i in range(len(patient_data_tensor)):
        predicted = model(patient_data_tensor[i])
        prediction = predicted.round()
        if prediction==1:
            print(f'Patient {i+1} has a heart disease')
        else:
            print(f'Patient {i+1} does not have a heart disease')
            
# ifunanyaScript