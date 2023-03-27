"""
Name:   MOHIT

Roll no. :  20323
"""
# Importing Libraries
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import tensorflow as tf 
import numpy as np
import torch 
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore') 

def line():
    print("\n------------------------------------")

print("The Download has been started for the given dataset")
line()

# Define the transform to normalize the input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the Olivetti faces dataset

class OlivettiFacesDataset(Dataset):
    def __init__(self, data, target, transform=transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.target[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Load the Olivetti faces dataset
olivetti_faces = fetch_olivetti_faces()

# Convert the data and target to PyTorch tensors
# data = torch.tensor(olivetti_faces.data.astype('float32'))
# target = torch.tensor(olivetti_faces.target)
data=olivetti_faces.data.astype('float32')
data=data.reshape(-1,64,64,1)
target=olivetti_faces.target

# Create a PyTorch dataset and data loader
dataset = OlivettiFacesDataset(data, target)


#creating architecture A


class CNNA(nn.Module):
    
    def __init__(self,number_of_classes=40,pool_size=3):

        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding="same")

        #Initialize the weights of Convolution Layer with He initialization for Leaky ReLU activation
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')

        #Create Batch Normalization 
        self.layer_1_batch_normalization = nn.BatchNorm2d(16)


        #Create Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU()
        self.pool_size=pool_size
        #Create Max Pooling 
        self.maxpool_1=nn.MaxPool2d(kernel_size=pool_size)
        


        #Create second Convolution Layer with input of layer 1 output channels (from first layer) and output of 32 channels
        self.layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity='leaky_relu')

        self.layer_2_batch_normalization = nn.BatchNorm2d(32)

        self.maxpool_2=nn.MaxPool2d(kernel_size=pool_size)



        #Create third Convolution Layer with input of layer 2 output channels (from second layer) and output of 64 channels
        self.layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity='leaky_relu')

        self.layer_3_batch_normalization = nn.BatchNorm2d(64)

        self.maxpool_3=nn.MaxPool2d(kernel_size=pool_size)

        #Create a fully connected layer for the CNN.
        self.fully_connected_layer_1 = nn.Linear(256, number_of_classes)
    
# forward function to predict input x.
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_1_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_1(x)
        
        x = self.layer_2(x)
        x = self.layer_2_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_2(x)


        x = self.layer_3(x)
        x = self.layer_3_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_3(x)
        
        #Flattening layer3's output and passing it into a fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer_1(x)
        return x
    

#**********************************************************************************************************************************************************************

class CNNB(nn.Module):
    
    def __init__(self,number_of_classes=40,pool_size=3):

        super().__init__()

        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding="same")
        self.layer_2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,padding="same")
        self.pool_size=pool_size

        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')

        #Create Batch Normalization 
        self.batch_normalization_1 = nn.BatchNorm2d(32)


        #Create Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU()

        #Create Max Pooling 
        self.maxpool_1=nn.MaxPool2d(kernel_size=pool_size)
        


        self.layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding="same")
        self.layer_4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding="same")

        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity='relu')

        #Create Batch Normalization 
        self.batch_normalization_2 = nn.BatchNorm2d(128)


        #Create Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU()

        #Create Max Pooling 
        self.maxpool_2=nn.MaxPool2d(kernel_size=pool_size)

        #Create a fully connected layer for the CNN.
        self.fully_connected_layer_1 = nn.Linear(6272, number_of_classes)
    
# forward function to predict input x.
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.batch_normalization_1(x)
        x = self.leaky_relu(x)
        x = self.maxpool_1(x)
        
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.batch_normalization_2(x)
        x = self.leaky_relu(x)
        x = self.maxpool_2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer_1(x)
        return x
        

#**********************************************************************************************************************************************************************


class CNNC(nn.Module):
    
    def __init__(self,number_of_classes=40,pool_size=3):

        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding="same")

        #Initialize the weights of Convolution Layer with He initialization for Leaky ReLU activation
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')

        #Create Batch Normalization 
        self.layer_1_batch_normalization = nn.BatchNorm2d(16)


        #Create Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU()
        self.pool_size=pool_size
        #Create Max Pooling 
        self.maxpool_1=nn.MaxPool2d(kernel_size=pool_size)
        


        #Create second Convolution Layer with input of layer 1 output channels (from first layer) and output of 32 channels
        self.layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity='leaky_relu')

        self.layer_2_batch_normalization = nn.BatchNorm2d(32)

        self.maxpool_2=nn.MaxPool2d(kernel_size=pool_size)



        #Create third Convolution Layer with input of layer 2 output channels (from second layer) and output of 64 channels
        self.layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity='leaky_relu')

        self.layer_3_batch_normalization = nn.BatchNorm2d(64)

        self.maxpool_3=nn.MaxPool2d(kernel_size=pool_size)

        #Create 2 fully connected layers for the CNN.
        self.fully_connected_layer_1 = nn.Linear(256, 4100)
        self.fully_connected_layer_2 = nn.Linear(4100, number_of_classes)
    
# forward function to predict input x.
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_1_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_1(x)
        
        x = self.layer_2(x)
        x = self.layer_2_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_2(x)


        x = self.layer_3(x)
        x = self.layer_3_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_3(x)
        
        #Flattening layer3's output and passing it into a fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer_1(x)
        x = self.fully_connected_layer_2(x)
        return x
    

#**********************************************************************************************************************************************************************
# function to initializd the model plot the accuracy charts and getting the results

def main(pool_size=3):
    print("The architecture for choice are:\nA. Conv-Pool-Conv-Pool-Conv-Pool-FC\nB. Conv-Conv-Pool-Conv-Conv-Pool-FC\nC. Conv-Pool-Conv-Pool-Conv-Pool-FC-FC ")
    cnn_arch=input("Enter the choice as either A, B or C: ")
    train_loader = DataLoader(dataset, batch_size=32)
    n_epochs=20
    if cnn_arch=="A":
            print("***************The Architecure of the CNN selected is A***************")
            model=CNNA(pool_size=pool_size)
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)#optimizer
            criterion = nn.CrossEntropyLoss()#loss
            training_acc_list=[]
            Accuracy_A={}
            N_train =400 
            for epoch in range(n_epochs):
                correct_train=0
                line()
                print(f"Epoch {epoch}")
                model.train()
                for x,y in tqdm(train_loader):
                    optimizer.zero_grad() #zero the gradient of the optimizer to adjust weights
                    y_pred=model(x)
                    loss=criterion(y_pred,y)
                    loss.backward() #backpropagation
                    optimizer.step() #update the parameters with respect to derivative calculated keeping the loss
                    correct_train += (torch.argmax(y_pred, 1) == y).sum().item()
                accuracy_train = round((correct_train / N_train)*100,2) #getting accuracy to only two decimal places
                print("Training accuracy: " + str(accuracy_train))
                training_acc_list.append(accuracy_train)
            plt.plot(training_acc_list)
            plt.grid(True)
            plt.savefig(f"Training Accuracy Plot for Architecture {cnn_arch} and pool size {pool_size} x {pool_size}.jpeg",bbox_inches="tight")
            plt.close()
            Accuracy_A[f"Training Accuracy for pool size {pool_size} x {pool_size}"]=max(training_acc_list)
    if cnn_arch=="B":
            print("***************The Architecure of the CNN selected is B***************")
            model=CNNB(pool_size=pool_size)
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)#optimizer
            criterion = nn.CrossEntropyLoss()#loss
            training_acc_list=[]
            Accuracy_B={}
            N_train = 400
            for epoch in range(n_epochs):
                correct_train=0
                line()
                print(f"Epoch {epoch}")
                model.train()
                for x,y in tqdm(train_loader):
                    optimizer.zero_grad() #zero the gradient of the optimizer to adjust weights
                    y_pred=model(x)
                    loss=criterion(y_pred,y)
                    loss.backward() #backpropagation
                    optimizer.step() #update the parameters with respect to derivative calculated keeping the loss
                    correct_train += (torch.argmax(y_pred, 1) == y).sum().item()
                accuracy_train = round((correct_train / N_train)*100,2) #getting accuracy to only two decimal places
                print("Training accuracy: " + str(accuracy_train))
                training_acc_list.append(accuracy_train)
            plt.plot(training_acc_list)
            plt.grid(True)
            plt.savefig(f"Training Accuracy Plot for Architecture {cnn_arch} and pool size {pool_size} x {pool_size}.jpeg",bbox_inches="tight")
            plt.close()
            Accuracy_B[f"Training Accuracy for pool size {pool_size} x {pool_size}"]=max(training_acc_list)
    if cnn_arch=="C":
            print("***************The Architecure of the CNN selected is C***************")
            model=CNNC(pool_size=pool_size)
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)#optimizer
            criterion = nn.CrossEntropyLoss()#loss
            val_acc_list=[]
            training_acc_list=[]
            Accuracy_C={}
            N_train = 400
            for epoch in range(n_epochs):
                correct_train=0
                line()
                print(f"Epoch {epoch}")
                model.train()
                for x,y in tqdm(train_loader):
                    optimizer.zero_grad() #zero the gradient of the optimizer to adjust weights
                    y_pred=model(x)
                    loss=criterion(y_pred,y)
                    loss.backward() #backpropagation
                    optimizer.step() #update the parameters with respect to derivative calculated keeping the loss
                    correct_train += (torch.argmax(y_pred, 1) == y).sum().item()
                accuracy_train = round((correct_train / N_train)*100,2) #getting accuracy to only two decimal places
                print("Training accuracy: " + str(accuracy_train))
                training_acc_list.append(accuracy_train)
            plt.plot(training_acc_list)
            plt.grid(True)
            plt.savefig(f"Training Accuracy Plot for Architecture {cnn_arch} and pool size {pool_size} x {pool_size}.jpeg",bbox_inches="tight")
            plt.close()
            Accuracy_C[f"Training Accuracy for pool size {pool_size} x {pool_size}"]=max(training_acc_list)

# *********************************************************************************************************************************************************************
#implementation 
while True:
    ans=input("Type y to proceed and q to Quit: ")
    if ans=="y":
        main(pool_size=3)
    elif ans=="q":
        break
line()
line()
print("The results depict the affect of change in pool size on the accurcy ")
class CNNA(nn.Module):
    
    def __init__(self,number_of_classes=40,pool_size=3):

        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding="same")

        #Initialize the weights of Convolution Layer with He initialization for Leaky ReLU activation
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')

        #Create Batch Normalization 
        self.layer_1_batch_normalization = nn.BatchNorm2d(16)


        #Create Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU()
        self.pool_size=pool_size
        #Create Max Pooling 
        self.maxpool_1=nn.MaxPool2d(kernel_size=pool_size)
        


        #Create second Convolution Layer with input of layer 1 output channels (from first layer) and output of 32 channels
        self.layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity='leaky_relu')

        self.layer_2_batch_normalization = nn.BatchNorm2d(32)

        self.maxpool_2=nn.MaxPool2d(kernel_size=pool_size)



        #Create third Convolution Layer with input of layer 2 output channels (from second layer) and output of 64 channels
        self.layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity='leaky_relu')

        self.layer_3_batch_normalization = nn.BatchNorm2d(64)

        self.maxpool_3=nn.MaxPool2d(kernel_size=pool_size)

        #Create a fully connected layer for the CNN.
        self.fully_connected_layer_1 = nn.Linear(64, number_of_classes)
    
# forward function to predict input x.
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_1_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_1(x)
        
        x = self.layer_2(x)
        x = self.layer_2_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_2(x)


        x = self.layer_3(x)
        x = self.layer_3_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_3(x)
        
        #Flattening layer3's output and passing it into a fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer_1(x)
        return x
    

#**********************************************************************************************************************************************************************

class CNNB(nn.Module):
    
    def __init__(self,number_of_classes=40,pool_size=3):

        super().__init__()

        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding="same")
        self.layer_2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,padding="same")
        self.pool_size=pool_size

        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')

        #Create Batch Normalization 
        self.batch_normalization_1 = nn.BatchNorm2d(32)


        #Create Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU()

        #Create Max Pooling 
        self.maxpool_1=nn.MaxPool2d(kernel_size=pool_size)
        


        self.layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding="same")
        self.layer_4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding="same")

        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity='relu')

        #Create Batch Normalization 
        self.batch_normalization_2 = nn.BatchNorm2d(128)


        #Create Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU()

        #Create Max Pooling 
        self.maxpool_2=nn.MaxPool2d(kernel_size=pool_size)

        #Create a fully connected layer for the CNN.
        self.fully_connected_layer_1 = nn.Linear(2048, number_of_classes)
    
# forward function to predict input x.
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.batch_normalization_1(x)
        x = self.leaky_relu(x)
        x = self.maxpool_1(x)
        
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.batch_normalization_2(x)
        x = self.leaky_relu(x)
        x = self.maxpool_2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer_1(x)
        return x
        

#**********************************************************************************************************************************************************************


class CNNC(nn.Module):
    
    def __init__(self,number_of_classes=40,pool_size=3):

        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding="same")

        #Initialize the weights of Convolution Layer with He initialization for Leaky ReLU activation
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')

        #Create Batch Normalization 
        self.layer_1_batch_normalization = nn.BatchNorm2d(16)


        #Create Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU()
        self.pool_size=pool_size
        #Create Max Pooling 
        self.maxpool_1=nn.MaxPool2d(kernel_size=pool_size)
        


        #Create second Convolution Layer with input of layer 1 output channels (from first layer) and output of 32 channels
        self.layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity='leaky_relu')

        self.layer_2_batch_normalization = nn.BatchNorm2d(32)

        self.maxpool_2=nn.MaxPool2d(kernel_size=pool_size)



        #Create third Convolution Layer with input of layer 2 output channels (from second layer) and output of 64 channels
        self.layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity='leaky_relu')

        self.layer_3_batch_normalization = nn.BatchNorm2d(64)

        self.maxpool_3=nn.MaxPool2d(kernel_size=pool_size)

        #Create 2 fully connected layers for the CNN.
        self.fully_connected_layer_1 = nn.Linear(64, 4100)
        self.fully_connected_layer_2 = nn.Linear(4100, number_of_classes)
    
# forward function to predict input x.
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_1_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_1(x)
        
        x = self.layer_2(x)
        x = self.layer_2_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_2(x)


        x = self.layer_3(x)
        x = self.layer_3_batch_normalization(x)
        x = self.leaky_relu(x)
        x = self.maxpool_3(x)
        
        #Flattening layer3's output and passing it into a fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer_1(x)
        x = self.fully_connected_layer_2(x)
        return x

while True:
    ans=input("Type y to proceed and q to Quit: ")
    if ans=="y":
        main(pool_size=4)
    elif ans=="q":
        break