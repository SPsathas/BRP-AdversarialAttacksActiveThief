import sys
import torch


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.sys_stream = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.log.write(message)
        self.sys_stream.write(message)

    def flush(self):
        pass


def accuracy(net, net_name, test_loader, cuda=False, idx=0):
    net.eval()

    total = 0.0
    correct = 0.0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data

        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        else:
            inputs = inputs.cpu()
            labels = labels.cpu()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        # if i % 100 == 0:
        #     count_p = 0
        #     x = (predicted == labels)
        #     for ii, c in enumerate(x, 0):
        #         if c == 1 and count_p < 10:
        #             count_p += 1
        #             print("accurate predicted:", predicted[ii], labels[ii])

    if net_name == 'netB':
        print('Accuracy of netB_%d: %.2f %%' % (idx, 100. * float(correct) / total))
    else:
        print('Accuracy of %s: %.2f %%' % (net_name, 100. * float(correct) / total))
    return 100. * float(correct) / total
