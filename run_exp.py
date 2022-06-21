import sys
import argparse

from common import Logger
from attack import MSAttack
from defense import MSDefense
import common as comm


def run_attack(args):
    msd = MSDefense(args)
    # Load Target Model for MNIST dataset
    msd.load(netv_path='saved_model/pretrained_net/net3conv_mnist.pth')
    # Load Target Model for FashionMNIST dataset
    # msd.load(netv_path='saved_model/pretrained_net/net3conv_fashionmnist.pth')

    msa = MSAttack(args, defense_obj=msd)
    msa.load()

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader)

    # Use the ActiveThief training method. Give strategy with input.
    msa.train_active_thief('saved_model/netS_mnist_temp.pth', label_only=False, strategy="random", epochs=10)

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader)

    # msa.attack("FGSM")
    # msa.attack("BIM")
    # msa.attack("CW")
    # msa.attack("PGD")


if __name__ == '__main__':
    sys.stdout = Logger('ms_attack.log', sys.stdout)

    args = argparse.ArgumentParser()
    args.add_argument('--cuda', default=False, action='store_true', help='using cuda')
    # Use for MNIST dataset
    args.add_argument('--dataset', type=str, default='MNIST')
    # Use for FashionMNIST dataset
    # args.add_argument('--dataset', type=str, default='FashionMNIST')
    args.add_argument('--num_class', type=int, default=10)

    args.add_argument('--epoch_b', type=int, default=20, help='for training net V')
    args.add_argument('--epoch_g', type=int, default=5, help='for training net S')

    args.add_argument('--lr', type=float, default=0.0001)
    args = args.parse_args()

    run_attack(args)


