import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_noshortcut(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out




class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class WResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(WResNet_cifar, self).__init__()
        self.in_planes = 16*k

        self.conv1 = nn.Conv2d(3, 16*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*k)
        self.layer1 = self._make_layer(block, 16*k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*k, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*k*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n])

def ResNet20_noshort():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet56():
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n])

def ResNet56_noshort():
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def WRN56_2():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 2)

def WRN56_4():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 4)

def WRN56_2_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 2)

def WRN56_4_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 4)


optimizer=optim.SGD()
criterion=nn.CrossEntropyLoss
net=ResNet20()
def train(trainloader, net, criterion, optimizer, use_cuda=True):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    return train_loss/total, 100 - 100.*correct/total