import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from advertorch.attacks import LinfBasicIterativeAttack, GradientSignAttack
from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack
from torch.utils.data import DataLoader

from active_thief import ActiveThief
from nets import Net2Conv, Net4Conv


class MSAttack(object):

    def __init__(self, args, defense_obj=None):
        self.args = args
        self.netS = None

        self.msd = defense_obj
        self.adversary = None

        self.train_loader = None
        self.test_loader = None

    def load(self, nets_path=None, net_type="Net2Conv"):
        """
        Loading nets and datasets
        """
        if self.args.dataset == 'MNIST':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.MNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.MNIST(root='dataset/', train=True, download=True, transform=transform)

            if net_type == "Net2Conv":
                SNet = Net2Conv
            else:
                SNet = Net4Conv

            if self.args.cuda:
                self.netS = SNet().cuda()
                map_location = lambda storage, loc: storage.cuda()
            else:
                self.netS = SNet().cpu()
                map_location = 'cpu'

            self.netS = nn.DataParallel(self.netS)
            if nets_path is not None:
                state_dict2 = torch.load(nets_path, map_location=map_location)
                self.netS.load_state_dict(state_dict2)

            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=2)

            print('Loading \'attack\' is done.')
        if self.args.dataset == "CIFAR10":
            print("Please extend the code to Cifar10 by yourself according to the existing MNIST example.")
        if self.args.dataset == 'FashionMNIST':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            test_set = torchvision.datasets.FashionMNIST(root='dataset/', train=False, download=True, transform=transform)
            train_set = torchvision.datasets.FashionMNIST(root='dataset/', train=True, download=True, transform=transform)

            if net_type == "Net2Conv":
                SNet = Net2Conv
            else:
                SNet = Net4Conv

            if self.args.cuda:
                self.netS = SNet().cuda()
                map_location = lambda storage, loc: storage.cuda()
            else:
                self.netS = SNet().cpu()
                map_location = 'cpu'

            self.netS = nn.DataParallel(self.netS)
            if nets_path is not None:
                state_dict2 = torch.load(nets_path, map_location=map_location)
                self.netS.load_state_dict(state_dict2)

            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, num_workers=2)
            self.train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=2)

            print('Loading \'attack\' is done.')

    @staticmethod
    def cross_entropy(q, p):
        return torch.mean(-torch.sum(p * torch.log(q), dim=1))

    def train_netS(self, path_s, path_g=None, data_type="REAL", label_only=False):
        if data_type == "REAL":
            self.train_netS_real(path_s, label_only)
        elif data_type == "synthetic":
            print("implement this according to the existing example")
        else:
            print("wrong data type")

    def train_netS_real(self, path_s, label_only):
        """
        Training the substitute net using real samples to query.
        """
        print("Starting training net S using real samples to query.")

        # optimizer_s = torch.optim.Adam(self.netS.parameters(), lr=self.args.lr)
        optimizer_s = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        self.netS.train()
        self.msd.netV.eval()

        for epoch in [0]:
            print("epoch: %d/%d" % (epoch + 1, self.args.epoch_g))

            for i, data in enumerate(self.train_loader, 0):
                # Updating netS
                self.netS.zero_grad()

                x_query, _ = data

                with torch.no_grad():
                    v_output = self.msd.netV(x_query)
                    v_output_p = F.softmax(v_output, dim=1)
                    _, v_predicted = torch.max(v_output_p, 1)

                s_output = self.netS(x_query.detach())
                s_prob = F.softmax(s_output, dim=1)

                if label_only:
                    loss_s = criterion(s_output, v_predicted)
                else:
                    loss_s = self.cross_entropy(s_prob, v_output_p)

                optimizer_s.zero_grad()
                loss_s.backward()
                optimizer_s.step()

                if i % 200 == 0:
                    print("batch idx:", i, "loss_s:", loss_s.detach().numpy())

        torch.save(self.netS.state_dict(), path_s)
        print("Finished training of netS")

    def train_active_thief(self, path_s, label_only, epochs, strategy="random", query=1000, a=0.1):
        """
            Training the substitute net using real samples to query.
        """
        print("Starting training net S using real samples to query.")
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root='dataset/', train=True, download=True, transform=transform)

        # optimizer_s = torch.optim.Adam(self.netS.parameters(), lr=self.args.lr)
        optimizer_s = torch.optim.RMSprop(self.netS.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        self.netS.train()
        self.msd.netV.eval()

        # Load the ActiveThief class
        AT = ActiveThief(data=train_set, batch_size=1000)
        seeding = True
        stop = False
        queries = 0
        epoch = 0
        while not stop and epoch < epochs:
            # Updating netS
            self.netS.zero_grad()

            self.netS.eval()
            dataloader = DataLoader(self.train_loader.dataset, batch_size=self.train_loader.dataset.data.shape[0], num_workers=12)
            # Query Substitute model
            for i, data in enumerate(dataloader, 0):
                if i % 100 == 0:
                    print("query substitute model")
                x_query, _ = data
                with torch.no_grad():
                    s_output = self.netS(x_query)
                    s_query = F.softmax(s_output, dim=1)

            if seeding:
                dataloader = AT.random_seeding(query)
                #dataloader = AT.cluster_seeding(s_query, query)
                AT.s_prob = s_query
                seeding = False
            else:
                dataloader = AT.subset_selection(s_query, strategy=strategy, labels=v_prop)

            query = 0
            v_prop = []
            for i, data in enumerate(dataloader, 0):

                x_query, _ = data

                with torch.no_grad():
                    v_output = self.msd.netV(x_query)
                    v_output_p = F.softmax(v_output, dim=1)
                    _, v_predicted = torch.max(v_output_p, 1)

                self.netS.train()
                s_output = self.netS(x_query.detach())
                s_prob = F.softmax(s_output, dim=1)
                _, s_predicted = torch.max(s_prob, 1)

                if label_only:
                    loss_s = criterion(s_output, v_predicted)
                else:
                    loss_s = self.cross_entropy(s_prob, v_output_p)

                optimizer_s.zero_grad()
                loss_s.backward()
                optimizer_s.step()

                v_prop.append(v_output_p)

                if i % 100 == 0:
                    print("Epoch", epoch, "batch idx:", i, "loss_s:", loss_s.detach().numpy())
                query = i+1

            queries += query
            epoch += 1
            stop = AT.stopping_criteria(a)

        torch.save(self.netS.state_dict(), path_s)
        print("Number of queries", queries)
        print("Finished training of netS")

    def get_adversary(self, method):
        if method == "FGSM":
            adversary = GradientSignAttack(
                self.netS, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.26, targeted=False)
        elif method == "BIM":
            adversary = LinfBasicIterativeAttack(
                self.netS, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
                nb_iter=200, eps_iter=0.02, clip_min=0.0, clip_max=1.0, targeted=False)
        elif method == "CW":
            adversary = CarliniWagnerL2Attack(
                self.netS, num_classes=10, learning_rate=0.45, binary_search_steps=10,
                max_iterations=12, targeted=False)
        elif method == "PGD":
            adversary = PGDAttack(
                self.netS, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=0.25, nb_iter=6, eps_iter=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
        else:
            # Using clean data samples
            adversary = None

        return adversary

    def create_adversary(self, method):
        self.adversary = self.get_adversary(method)
        self.netS.eval()

    def perturb(self, inputs, labels):
        if self.adversary is None:
            return inputs
        return self.adversary.perturb(inputs, labels)

    def attack(self, method="Clean"):
        self.create_adversary(method)

        correct = 0.0
        total = 0.0

        for data in self.test_loader:
            inputs, labels = data

            if self.args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

            adv_inputs = self.perturb(inputs, labels)
            with torch.no_grad():
                outputs = self.msd.netV(adv_inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum()
        print('Attack success rate of \'%s\': %.2f %%' % (method, (100 - 100. * float(correct) / total)))

    @staticmethod
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data = torch.where(m.weight.data > 0, m.weight.data, torch.zeros(m.weight.data.shape))


