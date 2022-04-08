from struct import pack_into
from scipy.io import loadmat
import torch
import numpy as np 
import matplotlib.pyplot as plt
import argparse 


parser = argparse.ArgumentParser()


parser.add_argument('--digit', default = 0, help = 'Number between 0 and 35. Index of Binary Digit to train on')
parser.add_argument("--q", default = 100, help="Number of hidden neurons")
parser.add_argument('--lr', default = 0.01, help = 'Learning rate for gradient ascent')
parser.add_argument('-bs', '--batch_size', default = 10, help = 'Number of input in batch')
parser.add_argument('--epochs', default = 1000, help = 'Number of training iterations')
parser.add_argument('-gi', '--gibbs_iter', default = 1000, help = 'Number of Gibbs iterations to generate images')
parser.add_argument('-ni', '--n_images', default = 1, help = 'Number of images to generate with trained RBM')


def lire_alpha_digit(label, device = 'cpu'):

    mat = loadmat('data/binaryalphadigs.mat')['dat']
    mat = mat[label]

    pixels = int(mat[0].shape[0] * mat[0].shape[1])
    n_data = mat.shape[0]

    X = np.zeros((n_data, pixels))

    for i in range(n_data):
        data = np.ndarray.flatten(mat[i])
        X[i, :] = data

    return torch.from_numpy(X).to(device).float()

class RBMClass(object) : 
    def __init__(self, input_size, output_size, device = 'cpu') : 
        self.W = torch.randn((input_size,output_size)).to(device)*0.01
        self.a = torch.zeros(input_size).to(device)
        self.b = torch.zeros(output_size).to(device)

def init_RBM(input_size, output_size, device):

    RBM = RBMClass(input_size, output_size, device)

    return RBM

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def entree_sortie_RBM(x, RBM):
    if len(x.size()) < 2 : 
        x = x.view(1,-1)
    return sigmoid(torch.matmul(x,RBM.W) + RBM.b)

def sortie_entree_RBM(h, RBM):
    
    return sigmoid(torch.matmul(h, RBM.W.T) + RBM.a)

def grad_W(x, v_1, RBM):   
    if len(x.size()) < 2 : 
        x = x.view(1,-1)
    grad = torch.matmul(x.T, entree_sortie_RBM(x, RBM)) - \
           torch.matmul(v_1.T, entree_sortie_RBM(v_1, RBM))
    return grad

def grad_a(x, v_1):
    return x - v_1

def grad_b(x, v_1, RBM):
    return entree_sortie_RBM(x, RBM) - entree_sortie_RBM(v_1, RBM)


def train_RBM(X, RBM, epochs, lr, batch_size):

    history = []

    for epoch in range(epochs):
        rdm_indices = torch.randperm(X.size(0))
        X_ = X[rdm_indices]
        batch_indices = torch.arange(start = 0, end = X.size(0), step = batch_size)

        reconstruction_error = 0

        for batch_indice in batch_indices:
            x = X_[batch_indice: batch_indice + batch_size]

            h_0 = entree_sortie_RBM(x, RBM)
            v_1 = sortie_entree_RBM(h_0, RBM)
            
            RBM.W += lr * grad_W(x, v_1, RBM)
            grad_a_ = grad_a(x, v_1)
            grad_b_ = grad_b(x, v_1, RBM)

            for idx in range(x.shape[0]) : 
                RBM.a += lr * grad_a_[idx]
                RBM.b += lr * grad_b_[idx]

            gen_h = entree_sortie_RBM(x, RBM)
            gen_x = sortie_entree_RBM(gen_h, RBM)

            reconstruction_error += torch.sum((x - gen_x) ** 2)
        history.append((reconstruction_error / X.shape[0]).item())
        if epoch % 50 == 0 : 
            print(f'EPOCH {epoch} - Reconstruction Error: {history[epoch]:0.4f}')
    return RBM, history

def generer_image_RBM(trained_RBM, gibbs_iters, n_images, plot = True):
    p, _ = trained_RBM.W.shape
    images = []

    for _ in range(n_images):
        x = torch.randn(p).to(trained_RBM.W.device)
        h = entree_sortie_RBM(x, trained_RBM)

        for _ in range(gibbs_iters):
            v = sortie_entree_RBM(h, trained_RBM)
            h = entree_sortie_RBM(v, trained_RBM)

        img = v.view( (20, 16)).detach().cpu().numpy()
        images.append(img)
        if plot : 
            plt.imshow(img)
            plt.show()

    return images

if __name__ == '__main__' : 
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available else 'cpu'

    X  = lire_alpha_digit(int(args.digit), device)
    p = X.shape[1]
    q = int(args.q)
    RBM = init_RBM(p, q, device)

    trained_RBM, history = train_RBM(X, RBM, int(args.epochs),
                                        float(args.lr), int(args.batch_size))
    generer_image_RBM(trained_RBM, int(args.gibbs_iter), int(args.n_images))