import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        # x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class PNet(nn.Module):

    def __init__(self, mp, kernel, cls_num, is_train=False):

        super(PNet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(mp, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, kernel, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, kernel, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, cls_num + 1, 1, 1)  # one extra cla for background cls
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a


class ONet(nn.Module):

    def __init__(self, linear, cls_num, is_train=False):

        super(ONet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),#25
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),#23

            ('conv2', nn.Conv2d(32, 64, 3, 1)),#11
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),#9

            ('conv3', nn.Conv2d(64, 64, 3, 1)),#4
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),#2

            ('conv4', nn.Conv2d(64, 128, 1, 1)),#1
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(128 * linear, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, cls_num + 1)  # one extra cla for background cls
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_mode = "CAR"
    if learning_mode == "CAR":
        Pnet = PNet((2, 2), (3, 3), 6).to(device)
        Onet = ONet(6, linear=1).to(device)
        p_width = p_height = 12
        o_width = o_height = 24
    elif learning_mode == "LPR":
        Pnet = PNet((2, 5), (3, 5), 6).to(device)
        Onet = ONet(2, linear=2 * 5).to(device)
        p_width = 12
        p_height = 47
        o_width = 24
        o_height = 94

    P_input = torch.Tensor(2, 3, p_width, p_height).to(device)
    P_offset, P_prob = Pnet(P_input)
    print('P_offset shape is', P_offset.shape)
    print('P_prob shape is', P_prob.shape)

    O_input = torch.Tensor(2, 3, o_width, o_height).to(device)
    O_offset, O_prob = Onet(O_input)
    print('O_offset shape is', O_offset.shape)
    print('O_prob shape is', O_prob.shape)

    
    # from torchsummary import summary
    # summary(Pnet, (3,12,47))
    # summary(Onet, (3,24,94))


