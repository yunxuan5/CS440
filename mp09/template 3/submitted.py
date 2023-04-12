# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        # raise NotImplementedError("You need to write this part!")
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.labels = []
        for file in data_files:
            data_set = unpickle(file)
            for i in range(len(data_set[b'data'])):
                self.data.append(data_set[b'data'][i])
                self.labels.append(data_set[b'labels'][i])

        # self.data_files = np.vstack(self.data_files).reshape(-1, 3, 32, 32)
        # self.data_files = self.data_files.transpose((0, 2, 3, 1))  # Convert to HWC format

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        # raise NotImplementedError("You need to write this part!")
        return len(self.data)

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        # raise NotImplementedError("You need to write this part!")
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # image, label = self.data_files[idx], self.target_transform[idx]
        img = self.data[idx]
        img = np.vstack(img).reshape(3, 32, 32)
        img = img.transpose((1, 2, 0))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """
    # raise NotImplementedError("You need to write this part!")
        
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    # raise NotImplementedError("You need to write this part!")
    # for file in data_files:
    datasets = CIFAR10(data_files, transform)
        # datasets.append(dataset)
    return datasets


"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    # raise NotImplementedError("You need to write this part!")
    dataloader = DataLoader(dataset, batch_size=loader_params["batch_size"], shuffle=loader_params["shuffle"])
    return dataloader


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################
        #load pretrained network
        file_path = 'resnet18.pt'
        file = torch.load(file_path)
        self.resnet18 = resnet18()
        self.resnet18.load_state_dict(file)
        # Freeze the convolutional backbone
        for param in self.resnet18.parameters():
            param.requires_grad = False

        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 8)   #8 classes
        # raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x = self.resnet18(x)
        return x
        # raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    # raise NotImplementedError("You need to write this part!")
    if optim_type == "Adam":
        optimizer = torch.optim.Adam(model_params, lr=hparams['lr'])
    elif optim_type == "SGD":
        optimizer = torch.optim.SGD(model_params, lr=hparams['lr'], momentum=0.9)
    return optimizer


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################
    model.train()
    for data, labels in train_dataloader:
        data, labels = data.to(device), labels.to(device)
        pred = model(data)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_dataloader:
            data, labels = data.to(device), labels.to(device)
            pred = model(data)
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    return test_acc
    # raise NotImplementedError("You need to write this part!")

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    # raise NotImplementedError("You need to write this part!")
    train_transform = get_preprocess_transform("train")
    test_transform = get_preprocess_transform("test")
    train_dataset = build_dataset(["cifar10_batches/data_batch_1",
                                      "cifar10_batches/data_batch_2",
                                      "cifar10_batches/data_batch_3",
                                      "cifar10_batches/data_batch_4",
                                      "cifar10_batches/data_batch_5"], transform=get_preprocess_transform("train"))
    test_dataset = build_dataset(["cifar10_batches/test_batch"], transform=get_preprocess_transform("train"))

    # train_dataset = build_dataset(data_files_train, train_transform)
    # test_dataset = build_dataset(data_files_test, test_transform)

    loader_params = {'batch_size': 4, 'shuffle': True}

    train_dataloader = build_dataloader(train_dataset, loader_params)
    test_dataloader = build_dataloader(test_dataset, loader_params)

    model = build_model(trained=True).to(device)
    optim_params = {'lr':0.001}
    optimizer = build_optimizer(optim_type="SGD", model_params=model.parameters(), hparams=optim_params)

    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 1

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(train_dataloader, model, loss_fn, optimizer)
        test_acc = test(test_dataloader, model)
        print(f"Test accuracy: {test_acc:.4f}")

    return model

