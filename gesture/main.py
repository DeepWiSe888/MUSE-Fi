import torch
import torch.nn as nn
import torch.optim as optim

import scipy.io
import torch
import torch.utils.data

import numpy as np

import pdb

num_epochs = 100

device = torch.device("cuda")

# Load the .mat files into numpy arrays
train_data = scipy.io.loadmat('../Data/gesture/gesture_train.mat')['data']
train_labels = scipy.io.loadmat('../Data/gesture/gesture_train.mat')['label']
test_data = scipy.io.loadmat('../Data/gesture/gesture_test.mat')['data']
test_labels = scipy.io.loadmat('../Data/gesture/gesture_test.mat')['label']

train_data = np.squeeze(train_data)
train_data = np.stack(train_data)

train_labels = np.squeeze(train_labels)
train_labels = np.stack(train_labels)

test_data = np.squeeze(test_data)
test_data = np.stack(test_data)

test_labels = np.squeeze(test_labels)
test_labels = np.stack(test_labels)

# Convert the numpy arrays to torch tensors
train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels).float()
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels).float()

train_labels = train_labels.squeeze()
test_labels = test_labels.squeeze()

train_data = train_data.unsqueeze(1)
test_data = test_data.unsqueeze(1)

# Create TensorDataset objects from the tensors
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

# Create DataLoader objects from the datasets
# You can specify the batch size, shuffle and other options here
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True)

class Net(nn.Module):
    def __init__(self, input_shape, n_class, n_gru_hidden_units, f_dropout_ratio):
        super(Net, self).__init__()
        self.conv2d = nn.Conv2d(1, 32, kernel_size=(5,5))
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(42112, 64)
        self.dropout = nn.Dropout(f_dropout_ratio)
        self.fc2 = nn.Linear(64, 64)
        self.gru = nn.GRU(64, n_gru_hidden_units, batch_first=True)
        self.fc3 = nn.Linear(n_gru_hidden_units, n_class)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = x.view(x.size(0), 1, -1) # Reshape for GRU
        x, _ = self.gru(x)

        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x

# Initialize the model, optimizer and loss function
model = Net(input_shape=(32, 1, 193), n_class=6, n_gru_hidden_units=128, f_dropout_ratio=0.5)
model = model.to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(num_epochs):
    print(epoch)
    loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # print(inputs.shape)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze()

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)

# Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 2)
        total += labels.size(0)
        correct += (predicted.squeeze() == labels.argmax(1)).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
