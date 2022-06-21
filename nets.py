import torch
import torch.nn as nn
import torch.nn.functional as F


class Net4Conv(nn.Module):
    def __init__(self):
        super(Net4Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net3Conv(nn.Module):
    def __init__(self):
        super(Net3Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2*2*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net2Conv(nn.Module):
    def __init__(self):
        super(Net2Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net5ConvCifar(nn.Module):
    def __init__(self):
        super(Net5ConvCifar, self).__init__()
        self.conv0 = nn.Conv2d(3, 20, 5, 1)
        self.conv1 = nn.Conv2d(20, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net4ConvCifar(nn.Module):
    def __init__(self):
        super(Net4ConvCifar, self).__init__()
        self.conv0 = nn.Conv2d(3, 20, 5, 1)
        self.conv1 = nn.Conv2d(20, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2*2*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net3ConvCifar(nn.Module):
    def __init__(self):
        super(Net3ConvCifar, self).__init__()
        self.conv0 = nn.Conv2d(3, 20, 5, 1)
        self.conv1 = nn.Conv2d(20, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetGenBase(nn.Module):

    def __init__(self, z_dim):
        super(NetGenBase, self).__init__()
        self.z_dim = z_dim
        self.decoder = None
        self.fc_dinput = None

    def decode(self, z) -> torch.Tensor:
        pass

    def forward(self, z):
        return self.decode(z)

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.z_dim)
        samples = self.decode(z)
        return samples

    def generate(self, z):
        return self.decode(z)


class NetGenMnist(NetGenBase):

    def __init__(self, z_dim=128):
        super(NetGenMnist, self).__init__(z_dim)

        dim = 5 * 4 * 4
        self.fc_dinput = nn.Linear(self.z_dim, dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 10, 5, stride=1),  # 5*4*4=>10*8*8
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(10, 10, 5, stride=4),  # 10*8*8=>10*33*33
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10, 1, 6, stride=1),  # 10*33*33=>1*28*28
            nn.BatchNorm2d(1),
            # nn.Tanh(),  # the value range (-1, 1)
            nn.ReLU(),
        )

    def decode(self, z) -> torch.Tensor:
        x = self.fc_dinput(z)
        x = x.view(x.shape[0], 5, 4, 4)
        # print("x:", x.shape)
        x = self.decoder(x)

        # x = torch.sign(x)
        # x = torch.relu(x)
        return x

