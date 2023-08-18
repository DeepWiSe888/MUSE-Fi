import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
import numpy as np
# Define the CNN architecture
from utils import load_MUSE_Fi_data,MUSEFiDataset
import dual

class CNN(nn.Module):
    def __init__(self, num_classes, height, weight):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.1)
        self.linear_input_size = 64 * 8 * 48
        self.fc1 = nn.Linear(self.linear_input_size, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        self.sigmod = nn.Softmax(dim=-1)


    def forward(self, x):
        a= x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(-1, self.linear_input_size)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmod(x)
        return x

def calculate_confusion_matrix(model, test_loader):
    model.eval()
    correct = 0
    confusion_matrix = np.zeros([6,6])
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch['data'], batch['label']
            data = Variable(data)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, predicted = torch.max(output.data, -1)
            _, target_ind = torch.max(target, -1)
            correct += (predicted == target_ind).sum().item()
            for ix in range(data.shape[0]):
                confusion_matrix[target_ind[ix], predicted[ix]] += 1
    for ix in range(6):
        confusion_matrix[ix,:] = confusion_matrix[ix,:] / np.sum(confusion_matrix[ix,:])
    return confusion_matrix

# Define the training and testing functions
def train(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    correct = 0.0
    for batch in train_loader:
        data,target = batch['data'], batch['label']
        data = Variable(data)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.shape[0]
        _, predicted = torch.max(output.data, -1)
        _, target_ind = torch.max(target, -1)
        correct += (predicted == target_ind).sum().item()
        # print('Loss: %f\n' % loss.item())
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100.0 * correct / len(train_loader.dataset)
    return train_loss, train_acc

def test(model, criterion, test_loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch['data'], batch['label']
            data = Variable(data)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.shape[0]
            _, predicted = torch.max(output.data, -1)
            _, target_ind = torch.max(target, -1)
            total += target.size(0)
            correct += (predicted == target_ind).sum().item()
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = 100 * correct / total
    return test_loss, test_acc

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--data', type=str, default='activity',
                    help='the dataset to run (default: my)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: True)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--batchsize', type=int, default=512,
                    help='batch_size (default: 512)')

num_class = 6
height = 32
width = 193



args = parser.parse_args()

X_train, Y_train, X_test, Y_test =  load_MUSE_Fi_data(args.data)

# Load the data
train_set = MUSEFiDataset(X=X_train, Y=Y_train)
loader_args = dict(batch_size=args.batchsize, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)

test_set = MUSEFiDataset(X=X_test, Y=Y_test)
test_loader = DataLoader(test_set, shuffle=True, **loader_args)

# Initialize the model, loss function, and optimizer
# model = CNN(num_class,height,width)

model = dual.Net()
if args.cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and test the model
for epoch in range(30):

    train_loss, train_acc = train(model, criterion, optimizer, train_loader)
    test_loss, test_acc = test(model, criterion, test_loader)
    print('Epoch {}, Train Loss: {:.4f}, Train Acc: {:.2f} Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(epoch+1, train_loss, train_acc, test_loss, test_acc))
    
