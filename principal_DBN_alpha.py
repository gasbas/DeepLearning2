import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse 

from principal_RBM_alpha import (
    init_RBM,
    entree_sortie_RBM,
    train_RBM,
    sortie_entree_RBM,
    lire_alpha_digit,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--digit",
    default=0,
    help="Number between 0 and 35. Index of Binary Digit to train on",
)
parser.add_argument("--q", default=100, help="Number of hidden neurons", type = int)
parser.add_argument('-n','--nlayers', default = 2, help = "Number of hidden layers", type = int)
parser.add_argument("--lr", default=0.1, help="Learning rate for gradient ascent", type = float)
parser.add_argument("-bs", "--batch_size", default=5, help="Number of input in batch", type = int)
parser.add_argument("--epochs", default=1000, help="Number of training iterations", type = int)
parser.add_argument(
    "-ni", "--n_images", default=1, help="Number of images to generate with trained RBM", type = int
)
parser.add_argument(
    "-gi",
    "--gibbs_iter",
    default=1000,
    help="Number of Gibbs iterations to generate images", type = int
)


def init_DNN(depth, sizes, device, random_state = None):
    if random_state is not None :
        torch.manual_seed(random_state)
    return {l: init_RBM(sizes[l][0], sizes[l][1], device) for l in range(depth)}


def pretrain_DNN(data, DNN, epochs, lr, batch_size, DBN = False):

    x = copy.deepcopy(data)
    trained_DNN = dict()

    for index, layer in DNN.items():

        
        if any([index < len(DNN)-1, DBN]) : 
            trained_DNN[index], _ = train_RBM(x, layer, epochs, lr, batch_size)
            x = entree_sortie_RBM(x, layer)
        else : 
            trained_DNN[index] = layer
    return trained_DNN


def entree_sortie_DBN(data_in, DNN):

    x = copy.deepcopy(data_in)

    for layer in list(DNN.values())[:-1]:
        x = entree_sortie_RBM(x, layer)

    return x


def sortie_entree_DBN(data_out, DNN):

    x = copy.deepcopy(data_out)

    for layer in reversed(list(DNN.values())[:-1]):
        x = sortie_entree_RBM(x, layer)

    return x


def generer_image_DBN(trained_DNN, gibbs_iters, n_images, plot=True):
    p, _ = trained_DNN[0].W.shape
    images = []
    for _ in range(n_images):
        x = torch.randn(p).to(trained_DNN[0].W.device)
        h = entree_sortie_DBN(x, trained_DNN)
        for _ in range(gibbs_iters):
            v = sortie_entree_DBN(h, trained_DNN)
            h = entree_sortie_DBN(v, trained_DNN)
        img = v.view((20, 16)).detach().cpu().numpy()
        images.append(img)
        if plot:
            plt.imshow(img)
            plt.show()
    return images


if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available else "cpu"

    X = lire_alpha_digit(int(args.digit), device)
    sizes = [(X.size(1), args.q)]
    for j in range(args.nlayers) : 
        sizes.append((args.q, args.q))

    DNN = init_DNN(len(sizes), sizes, device)

    trained_DNN = pretrain_DNN(
         X, DNN, args.epochs, args.lr, args.batch_size
     )
    generer_image_DBN(trained_DNN, args.gibbs_iter, args.n_images)
