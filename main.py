import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline 

# load filenames for human and dog images
human_files = np.array(glob("/data/human_images/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))

                              
# Detect human faces
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# Human face detector: returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# Assess the Human Face Detector
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

# define a function to calculate the ratio
def face_images_ratio(file_paths):
    face_images = 0
    for file_path in file_paths:
        if (face_detector(file_path) == True):
            face_images += 1
    return face_images / len(file_paths)

print(f"The percentage of the detected human faces in human_files is {face_images_ratio(human_files_short):.1%}.")
print(f"The percentage of the detected human faces in dog_files is {face_images_ratio(dog_files_short):.1%}.")




# Detect Dog faces using pre-trained VGG-16 network
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


from PIL import Image
import torchvision.transforms as transforms

print(use_cuda)

device = torch.device('cuda' if use_cuda else 'cpu')

print(device)


def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    #img_3D = cv2.imread(img_path)
    img_3D = Image.open(img_path)
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor()])
    img_3D_Tensor = transform(img_3D)
    img_4D_Tensor = torch.unsqueeze(img_3D_Tensor, 0)
    
    img_4D_Tensor = img_4D_Tensor.to(device)
    
    
    output = VGG16(img_4D_Tensor)
    max_val, index = torch.max(output.view(1,-1), dim = 1)
    
    return index 


    # Assess dog face detector: returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    if (VGG16_predict(img_path) > 150 and VGG16_predict(img_path) < 269):
        return True
    else:
        return False


# define a function to calculate the ratio
def dog_images_ratio(file_paths):
    dog_images = 0
    for file_path in file_paths:
        if (dog_detector(file_path) == True):
            dog_images += 1
    return dog_images / len(file_paths)

print(f"The percentage of the detected dogs in human_files is {dog_images_ratio(human_files_short):.1%}.")
print(f"The percentage of the detected dogs in dog_files is {dog_images_ratio(dog_files_short):.1%}.")


#----------------------------
# Creat a CNN from scratch
#----------------------------
import os
from torchvision import datasets

# load training data
batch_sizes = 64
train_transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.CenterCrop((224,224)),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
train_dataset = datasets.ImageFolder('/data/dog_images/train', transform = train_transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_sizes, shuffle = True)

# load validation data
#batch_sizes = 64
valid_transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.CenterCrop((224,224)),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
valid_dataset = datasets.ImageFolder('/data/dog_images/valid', transform = valid_transform)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_sizes, shuffle = True)

# load test data
test_batch_sizes = 1
test_transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.CenterCrop((224,224)),
                                     transforms.ToTensor()])
test_dataset = datasets.ImageFolder('/data/dog_images/test', transform = test_transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_sizes, shuffle = True)

loaders_scratch = {}
loaders_scratch['train'] = trainloader
loaders_scratch['valid'] = validloader
loaders_scratch['test'] = testloader


import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1, bias = False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1, bias = False)
        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False)
        self.conv6 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1, bias = False)
        #self.conv7 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = False)
        #self.conv8 = nn.Conv2d(512, 1024, kernel_size = 3, padding = 1, bias = False)
        
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #self.bn7 = nn.BatchNorm2d(self.conv7.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #self.bn8 = nn.BatchNorm2d(self.conv8.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        
        
        self.maxPool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)
        
        #self.fc1 = nn.Linear(7*7*self.conv8.out_channels, 1024*10)
        #self.fc2 = nn.Linear(self.fc1.out_features, 1024)
        #self.fc3 = nn.Linear(1024,133)
        
        self.fc1 = nn.Linear(7*7*self.conv6.out_channels, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 133)
        
        
    
    def forward(self, x):
        ## Define forward behavior
        hidden_layer1_in = self.conv1(x)
        hidden_layer1_bn = self.bn1(hidden_layer1_in)
        hideen_layer1_out = self.maxPool(F.relu(hidden_layer1_bn))
        hideen_layer1_out = self.dropout(hideen_layer1_out)
        
        hidden_layer2_in = self.conv2(hideen_layer1_out)
        hidden_layer2_bn = self.bn2(hidden_layer2_in)
        hideen_layer2_out = self.maxPool(F.relu(hidden_layer2_bn))
        hideen_layer2_out = self.dropout(hideen_layer2_out)
        
        hidden_layer3_in = self.conv3(hideen_layer2_out)
        hidden_layer3_bn = self.bn3(hidden_layer3_in)
        hideen_layer3_out = F.relu(hidden_layer3_bn)
        hideen_layer3_out = self.dropout(hideen_layer3_out)
        
        hidden_layer4_in = self.conv4(hideen_layer3_out)
        hidden_layer4_bn = self.bn4(hidden_layer4_in)
        hideen_layer4_out = self.maxPool(F.relu(hidden_layer4_bn))
        hideen_layer4_out = self.dropout(hideen_layer4_out)
        
        hidden_layer5_in = self.conv5(hideen_layer4_out)
        hidden_layer5_bn = self.bn5(hidden_layer5_in)
        hideen_layer5_out = self.maxPool(F.relu(hidden_layer5_bn))
        hideen_layer5_out = self.dropout(hideen_layer5_out)
        
        hidden_layer6_in = self.conv6(hideen_layer5_out)
        hidden_layer6_bn = self.bn6(hidden_layer6_in)
        hideen_layer6_out = self.maxPool(F.relu(hidden_layer6_bn))
        hideen_layer6_out = self.dropout(hideen_layer6_out)
        

        
        ## classifer layers
        classifier_layer1 = F.relu(self.fc1(hideen_layer6_out.view((-1, 7*7*self.conv6.out_channels))))
        classifier_layer1 = self.dropout(classifier_layer1)
        
        #classifier_layer2 = F.relu(self.fc2(classifier_layer1))
        #classifier_layer2 = self.dropout(classifier_layer2)
        
        classifier_layer2 = self.fc2(classifier_layer1)
        
        return classifier_layer2

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
    
print(model_scratch)

import torch.optim as optim

### loss function
criterion_scratch = nn.CrossEntropyLoss()

### optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr = 0.01)


## Train
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    train_loss_epochs = []
    valid_loss_epochs = []
    
    for epoch in range(1, n_epochs+1):
        start = time.time()
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            #train_loss += loss.item() # accumulate the loss over the batch
            
        #train_loss = train_loss / (len(data)*(batch_idx+1)) # average the loss
        #train_loss_epochs.append(train_loss)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                output = model(data)
                loss = criterion(output, target)
                
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                #valid_loss += loss.item()
                
        #valid_loss = valid_loss / (len(data)*(batch_idx+1)) # average the loss
        #valid_loss_epochs.append(valid_loss)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print("Valid_loss decreases: saving the model...")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
        print(f"It takes {time.time() - start:.2f} seconds")
            
    # return trained model
    return model


# train the model
model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')


# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))


def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        #test_loss += loss.item()
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
    
    #test_loss = test_loss / ((batch_idx+1)*len(data))
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)




#-------------------------------------
# Creat a CNN using transfer learning
#-------------------------------------

# load the data 
batch_sizes = 64
train_transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.CenterCrop((224,224)),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
train_dataset = datasets.ImageFolder('/data/dog_images/train', transform = train_transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_sizes, shuffle = True)

# load validation data
#batch_sizes = 64
valid_transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.CenterCrop((224,224)),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
valid_dataset = datasets.ImageFolder('/data/dog_images/valid', transform = valid_transform)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_sizes, shuffle = True)

# load test data
test_transform = transforms.Compose([transforms.Resize((256,256)),
                                     transforms.CenterCrop((224,224)),
                                     transforms.ToTensor()])
test_dataset = datasets.ImageFolder('/data/dog_images/test', transform = test_transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_sizes, shuffle = True)

loaders_transfer = {}
loaders_transfer['train'] = trainloader
loaders_transfer['valid'] = validloader
loaders_transfer['test'] = testloader

data_transfer = {}
data_transfer['train'] = train_dataset
data_transfer['valid'] = valid_dataset
data_transfer['test'] = test_dataset


import torchvision.models as models
import torch.nn as nn

## Pre-trained net architecture
model_transfer = models.resnet18(pretrained = True)
print(model_transfer)

#classifier = nn.Sequential(nn.Linear(model_transfer.fc.in_features, 1000),
                           #nn.ReLU(),
                           #nn.Linear(1000,133)) # for resnet50

classifier = nn.Sequential(nn.Linear(model_transfer.fc.in_features, 133))
model_transfer.fc = classifier


if use_cuda:
    model_transfer = model_transfer.cuda()


## Loss function and optimizer
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr = 0.001)

## train
n_epochs = 100
model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))

# test
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)



#### prediction
# predict gog breed with the model

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].classes]

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    img_3D = Image.open(img_path)
    
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor()]) 
    img_3D_Tensor = transform(img_3D)
    img_4D_Tensor = torch.unsqueeze(img_3D_Tensor, 0)
    
    img_4D_Tensor = img_4D_Tensor.to(device)
    
    #output = model_transfer(img_4D_Tensor) # if predicted by pre-trained CNN model
    output = model_scratch(img_4D_Tensor) # if predicted by pre-trained CNN model
    max_val, index = torch.max(output.view(1,-1), dim = 1)
    
    return class_names[index]


def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    if (dog_detector(img_path) == True):
        breed = predict_breed_transfer(img_path)
        print(f"Hello, dog! Your breed is {breed}!")
    elif (face_detector(img_path) == True):
        breed = predict_breed_transfer(img_path)
        print(f"Hello, human! You look like a {breed}!")
    else:
        print("The image contains neither dog nor hunman!")
    img_3D = Image.open(img_path)
    plt.imshow(img_3D)
    plt.show()


# User images
UserImage_files = np.array(glob('UserImages/*'))
#print(myImage_files)
#for file in np.hstack((human_files[:3], dog_files[:3])):
for file in np.hstack((human_files[:3], UserImage_files[:])):
    run_app(file)