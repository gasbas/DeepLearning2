from principal_RBM_alpha import * 
from principal_DBN_alpha import * 
import torch
import numpy as np
from time import time
from tqdm import tqdm
import torchvision
import argparse
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--digit",
    default=0,
    help="Number between 0 and 35. Index of Binary Digit to train on",
)
parser.add_argument("--q", default=100, help="Number of hidden neurons", type = int)
parser.add_argument('-n','--nlayers', default = 2, type = int)
parser.add_argument("--lr", default=0.1, help="Learning rate for gradient ascent", type = float)
parser.add_argument("-bs", "--batch_size", default=512, help="Number of input in batch", type = int)
parser.add_argument("--epochs_retro", default=200, help="Number of training iterations for retropropagation", type = int)
parser.add_argument("--pretrain", default = "store_true", help="Whether to use pretraining")
parser.add_argument("--epochs_pretrain", default=100, help="Number of training iterations for pretraining", type = int)
parser.add_argument("--val_size", default=0, help="Fraction of training data to use as validation", type = float)
parser.add_argument("--n_train", default=60000, help="Number of images to use as training data", type = int)
parser.add_argument("--esr", default=30, help="Number of early stopping rounds", type = int)


def lire_mnist(device = 'cpu', val_size = 0.05, n_images = 60000)  : 
    train_data = torchvision.datasets.MNIST('data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                 ]))
    test_data = torchvision.datasets.MNIST('data/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                 ]))
    x_train, y_train = train_data.data.view(-1,28*28) / 255, train_data.targets
    x_test, y_test = test_data.data.view(-1,28*28) / 255, test_data.targets
    x_train, y_train = x_train[:n_images], y_train[:n_images]
    
    x_train[x_train < 0.5] = 0 
    x_train[x_train>=0.5] = 1
    x_test[x_test<0.5] = 0
    x_test[x_test>=0.5] = 1
    
    if val_size > 0 : 
        val_indices = torch.randperm(x_train.size(0))[: int(x_train.size(0) * val_size)]
        train_indices = [i for i in range(x_train.size(0)) if i not in val_indices]
        x_val, y_val = x_train[val_indices], y_train[val_indices]
        x_train, y_train = x_train[train_indices], y_train[train_indices]

        return x_train.float().to(device), y_train.long().to(device), x_val.float().to(device), y_val.long().to(device), x_test.float().to(device), y_test.long().to(device)
    else : 
        return x_train.float().to(device), y_train.long().to(device), None, None, x_test.float().to(device), y_test.long().to(device)
     
def one_hot_encode(y, num_classes = 10) : 
    
    y_ = torch.zeros(y.size(0), num_classes).to(y.device)
    indices = torch.arange(y.size(0))
    y_[indices,y] = 1
    return y_

def calcul_softmax(x, RBM):
    z = torch.mm(x, RBM.W) + RBM.b
    return torch.exp(z)/torch.sum(torch.exp(z), dim = 1, keepdims = True)

def entree_sortie_reseau(x, DNN):
    depth = len(DNN)
    Z = []
    Z.append(x)
    for l in range(depth):
        if l == depth - 1:
            Z.append(calcul_softmax(x, DNN[l]))

        else:
            x = entree_sortie_RBM(x, DNN[l])
            Z.append(x)
    
    return Z

def retropropagation(X, y, DNN, epochs, lr, batch_size, x_val = None, y_val = None, esr = 20):

    depth = len(DNN)
    history = []
    n_data = X.size(0)
    best_acc_val = 0
    e=0
    pbar=  tqdm(range(epochs))
    for epoch in pbar:

        rdm_indices = torch.randperm(n_data)
        X_ = X[rdm_indices]
        y_shuffled = y[rdm_indices]
        batch_indices = torch.arange(start=0, end=n_data, step=batch_size)
        loss = 0
    
        for i in batch_indices:
            output = entree_sortie_reseau(X_[i : i + batch_size], DNN)
            y_pred = output[-1]
            y_ = one_hot_encode(y_shuffled[i : i + batch_size])
                
            loss += -torch.sum(y_ * torch.log(y_pred+1e-8)) 
            
            for l in range(depth, 0, -1):

                if l == depth :
                    c = y_pred - y_
                else:
                    c = c * output[l] * (1 - output[l])

                grad_W = torch.mm(output[l-1].T, c)
                grad_b = c
                
                DNN[l-1].W -= lr * grad_W / batch_size
                DNN[l-1].b -= lr * grad_b.sum(dim = 0) / batch_size
                
                c = torch.mm(DNN[l-1].W, c.T).T
        
        
        if all([x_val is not None, y_val is not None]) : 
            accuracy_val = test_DNN(x_val,y_val, DNN)
            
            if accuracy_val > best_acc_val : 
                best_acc_val = accuracy_val
                e = 0
                best_DNN = copy.deepcopy(DNN)
            else : 
                e+=1 
                if e == esr : 
                    return best_DNN, history
        else : 
            accuracy_val = test_DNN(X,y, DNN)
            
        pbar.set_description(f'Epoch {epoch+1} | loss : {loss/X.size(0):0.3f} | accuracy {accuracy_val: 0.2f}')
        history.append((loss / X.size(0)).item())

    return DNN, history

def test_DNN(images, labels, DNN):
        
    y_pred = torch.argmax(entree_sortie_reseau(images, DNN)[-1], dim = 1)
        
            
    return (y_pred.eq(labels).sum() / y_pred.size(0)).item()
        
if __name__ == '__main__' : 

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()

    x_train, y_train, x_val, y_val, x_test, y_test = lire_mnist(device = device, val_size = args.val_size, n_images = args.n_train)
    print(f'Number of training images : {x_train.size(0)}')
    if x_val is not None: 
        print(f'Number of validation images : {x_val.size(0)}')
    else : 
        print(f'Number of validation images : {0}')

    sizes = [(x_train.size(1), args.q)]
    for j in range(args.nlayers) : 
        sizes.append((args.q, args.q))
    sizes.append((args.q, 10))
    print(f'Using {args.nlayers} layers with {args.q} neurons')

    DNN = init_DNN(len(sizes), sizes, device)
    if args.pretrain : 
        print(f'Pretraining for {args.epochs_pretrain} epochs')
        DNN = pretrain_DNN(x_train, DNN, args.epochs_pretrain,
                                      args.lr, args.batch_size)
    if x_val is not None : 
        print(f'Training until validation accruacy stopped increasing for {args.esr} rounds')
    else :
        print(f'Training for {args.epochs_retro} epochs')
    DNN, history = retropropagation(x_train, y_train, DNN, args.epochs_retro, args.lr, args.batch_size, x_val,y_val, esr = args.esr)

    accuracy_test = test_DNN(x_test,y_test, DNN)
    accuracy_train = test_DNN(x_train, y_train, DNN)
    accuracy_val = test_DNN(x_val, y_val, DNN)

    print(f'Accuracy Train : {accuracy_train:0.3f}')
    print(f'Accuracy Val : {accuracy_val:0.3f}')
    print(f'Accuracy Test : {accuracy_test:0.3f}')