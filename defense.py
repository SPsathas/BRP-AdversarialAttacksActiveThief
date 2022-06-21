import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from nets import Net3Conv


class MSDefense(object):

    def __init__(self, args, attack_obj=None):
        super(MSDefense, self).__init__()
        self.args = args
        self.msa = attack_obj

        self.netV = None
        self.netB_list = []

        self.test_loader = None
        self.train_loader = None

    def load(self, netv_path=None, netb_plist=None):
        """
        Loading nets and datasets
        """
        if self.args.dataset == 'MNIST':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.MNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.MNIST(root='dataset/', train=True, download=True, transform=transform)

            data_list = [i for i in range(6000, 8000)]
            sampler = torch.utils.data.sampler.SubsetRandomSampler(data_list)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, sampler=sampler, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

            if self.args.cuda:
                self.netV = Net3Conv().cuda()
                if netb_plist is None:
                    self.netB_list.append(Net3Conv().cuda())  # for training NetB
                else:
                    # for defense evaluation
                    self.netB_list = []
                    for c in range(len(netb_plist)):
                        self.netB_list.append(Net3Conv().cuda())
                map_location = lambda storage, loc: storage.cuda()
            else:
                self.netV = Net3Conv().cpu()
                if netb_plist is None:
                    self.netB_list.append(Net3Conv().cpu())  # for training NetB
                else:
                    # for defense evaluation
                    self.netB_list = []
                    for c in range(len(netb_plist)):
                        self.netB_list.append(Net3Conv().cpu())
                map_location = 'cpu'

            self.netV = nn.DataParallel(self.netV)
            if netv_path is not None:
                state_dict = torch.load(netv_path, map_location=map_location)
                self.netV.load_state_dict(state_dict)
            self.netV.eval()

            for i in range(len(self.netB_list)):
                self.netB_list[i] = nn.DataParallel(self.netB_list[i])

            if netb_plist is not None:
                for i, path in enumerate(netb_plist, 0):
                    state_dict = torch.load(path, map_location=map_location)
                    self.netB_list[i].load_state_dict(state_dict)
        if self.args.dataset == 'CIFAR10':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.CIFAR10(root='dataset/', train=False, download=True,
                                                         transform=transform)
            train_set = torchvision.datasets.CIFAR10(root='dataset/', train=True, download=True,
                                                          transform=transform)

            data_list = [i for i in range(6000, 8000)]
            sampler = torch.utils.data.sampler.SubsetRandomSampler(data_list)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, sampler=sampler, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

            if self.args.cuda:
                self.netV = Net3Conv().cuda()
                if netb_plist is None:
                    self.netB_list.append(Net3Conv().cuda())  # for training NetB
                else:
                    # for defense evaluation
                    self.netB_list = []
                    for c in range(len(netb_plist)):
                        self.netB_list.append(Net3Conv().cuda())
                map_location = lambda storage, loc: storage.cuda()
            else:
                self.netV = Net3Conv().cpu()
                if netb_plist is None:
                    self.netB_list.append(Net3Conv().cpu())  # for training NetB
                else:
                    # for defense evaluation
                    self.netB_list = []
                    for c in range(len(netb_plist)):
                        self.netB_list.append(Net3Conv().cpu())
                map_location = 'cpu'

            self.netV = nn.DataParallel(self.netV)

            # if netv_path is not None:
            #     state_dict = torch.load(netv_path, map_location=map_location)
            #     self.netV.load_state_dict(state_dict)
            self.netV.train()
            self.netV.zero_grad()
            optimizer_s = torch.optim.RMSprop(self.netV.parameters(), lr=self.args.lr)
            for i, (data, label) in enumerate(self.train_loader, 0):
                predicted = self.netV(data.detach())
                prob = F.softmax(predicted, dim=1)
                loss = F.cross_entropy(prob, label)

                optimizer_s.zero_grad()
                loss.backward()
                optimizer_s.step()

                if i % 200 == 0:
                    print("batch idx:", i, "loss_s:", loss.detach().numpy())

            self.netV.eval()
            torch.save(self.netV.state_dict(), 'saved_model/pretrained_net/net3conv_cifar10.pth')

            for i in range(len(self.netB_list)):
                self.netB_list[i] = nn.DataParallel(self.netB_list[i])

            if netb_plist is not None:
                for i, path in enumerate(netb_plist, 0):
                    state_dict = torch.load(path, map_location=map_location)
                    self.netB_list[i].load_state_dict(state_dict)
        if self.args.dataset == 'FashionMNIST':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.FashionMNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.FashionMNIST(root='dataset/', train=True, download=True, transform=transform)

            data_list = [i for i in range(6000, 8000)]
            sampler = torch.utils.data.sampler.SubsetRandomSampler(data_list)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, sampler=sampler, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=2)

            if self.args.cuda:
                self.netV = Net3Conv().cuda()
                if netb_plist is None:
                    self.netB_list.append(Net3Conv().cuda())  # for training NetB
                else:
                    # for defense evaluation
                    self.netB_list = []
                    for c in range(len(netb_plist)):
                        self.netB_list.append(Net3Conv().cuda())
                map_location = lambda storage, loc: storage.cuda()
            else:
                self.netV = Net3Conv().cpu()
                if netb_plist is None:
                    self.netB_list.append(Net3Conv().cpu())  # for training NetB
                else:
                    # for defense evaluation
                    self.netB_list = []
                    for c in range(len(netb_plist)):
                        self.netB_list.append(Net3Conv().cpu())
                map_location = 'cpu'

            self.netV = nn.DataParallel(self.netV)

            if netv_path is not None:
                state_dict = torch.load(netv_path, map_location=map_location)
                self.netV.load_state_dict(state_dict)
            # self.netV.train()
            # self.netV.zero_grad()
            # optimizer_s = torch.optim.RMSprop(self.netV.parameters(), lr=self.args.lr)
            # for i, (data, label) in enumerate(self.train_loader, 0):
            #     predicted = self.netV(data.detach())
            #     prob = F.softmax(predicted, dim=1)
            #     loss = F.cross_entropy(prob, label)
            #
            #     optimizer_s.zero_grad()
            #     loss.backward()
            #     optimizer_s.step()
            #
            #     if i % 200 == 0:
            #         print("batch idx:", i, "loss_s:", loss.detach().numpy())

            self.netV.eval()
            # torch.save(self.netV.state_dict(), 'saved_model/pretrained_net/net3conv_fashionmnist.pth')

            for i in range(len(self.netB_list)):
                self.netB_list[i] = nn.DataParallel(self.netB_list[i])

            if netb_plist is not None:
                for i, path in enumerate(netb_plist, 0):
                    state_dict = torch.load(path, map_location=map_location)
                    self.netB_list[i].load_state_dict(state_dict)

        print("Loading \'defense\' is done.")

    def train_netV(self, save_path):
        print("Starting training the victim net V")

        optimizer = torch.optim.Adam(self.netV.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.netV.train()
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.args.epoch_b):
            print("epoch: %d / %d" % (epoch+1, self.args.epoch_b))
            for idx, data in enumerate(self.train_loader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self.netV(inputs)

                loss = criterion(outputs, labels)
                # loss = self.cross_entropy(F.softmax(outputs, dim=1), F.one_hot(labels, 10).float())

                if idx % 100 == 0:
                    print("loss:", loss)

                loss.backward()
                optimizer.step()

        torch.save(self.netV.state_dict(), save_path)
        print("Finished training of netV")

