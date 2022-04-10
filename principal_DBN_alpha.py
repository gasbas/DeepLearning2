import sys
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch

from principal_RBM_alpha import (
    init_RBM,
    entree_sortie_RBM,
    train_RBM,
    sortie_entree_RBM,
    lire_alpha_digit,
)


parser.add_argument(
    "--digit",
    default=0,
    help="Number between 0 and 35. Index of Binary Digit to train on",
)
parser.add_argument("--q", default=100, help="Number of hidden neurons")
parser.add_argument("--lr", default=0.01, help="Learning rate for gradient ascent")
parser.add_argument("-bs", "--batch_size", default=10, help="Number of input in batch")
parser.add_argument("--epochs", default=1000, help="Number of training iterations")
parser.add_argument(
    "-gi",
    "--gibbs_iter",
    default=1000,
    help="Number of Gibbs iterations to generate images",
)
parser.add_argument(
    "-ni", "--n_images", default=1, help="Number of images to generate with trained RBM"
)


def init_DNN(depth, sizes, device):

    return {l: init_RBM(sizes[l][0], sizes[l][1], device) for l in range(depth)}


def pretrain_DNN(data, DNN, epochs, lr, batch_size):

    x = copy.deepcopy(data)
    trained_DNN = dict()

    for index, layer in DNN.items():
        trained_DNN[index], _ = train_RBM(x, layer, epochs, lr, batch_size)
        x = entree_sortie_RBM(x, layer)
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
    print("DONE")
    # args = parser.parse_args()
    # device = torch.device("cuda") if torch.cuda.is_available else "cpu"

    # X = lire_alpha_digit(int(args.digit), device)
    # depth = 5
    # sizes = [(320, 100), (100, 100), (100, 100),(100, 100),(100, 100)]
    # DNN = init_DNN(depth, sizes, device)

    # trained_DNN, history = pretrain_DNN(
    #     X, RBM, int(args.epochs), float(args.lr), int(args.batch_size)
    # )
    # generer_image_DBN(trained_DNN, int(args.gibbs_iter), int(args.n_images))
