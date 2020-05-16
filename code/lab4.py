from __future__ import print_function
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision import datasets, transforms
from models.ResNet import *
import time
import os

# Training settings
parser = argparse.ArgumentParser(description='HPC lab2 example')
parser.add_argument('--title', type=str, default='NO TITLE',
                    help="name of this process")
parser.add_argument('--data', type=str, default='./data',
                    help="folder path where data is located.")
parser.add_argument('--cuda', type=int, default=0, choices={0, 1},
                    help="sets the usage of cuda gpu training.")
parser.add_argument('--workers', type=int, default=2,
                    help="sets number of workers used in DataLoader")
parser.add_argument('--optim', type=str, default='sgd', choices={"sgd", "nesterov", "adagrad", "adadelta", "adam"},
                    help="optimers include sgd,")
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size per gpu for training (default: 128)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs to train (default: 5)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--disable_parallel', action='store_true',
                    help='disable multi-gpu data parallel')
parser.add_argument('--gpu_count', type=int, default=1,
                    help='number of gpus used (default: 1)')
args = parser.parse_args()
print(args)

if torch.cuda.is_available() and args.cuda == 0 and args.gpu_count > 0:  # CPU or GPU
    use_gpu = True
else:
    use_gpu = False

gpu_count = torch.cuda.device_count() if args.gpu_count > torch.cuda.device_count() else args.gpu_count

if use_gpu:
    batch_size = args.batch_size * gpu_count
else:
    batch_size = args.batch_size

torch.manual_seed(1)

# Load and transform data
data_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # default is p=0.5
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])
if not os.path.isdir("./data"):
    os.mkdir("./data")
trainset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=data_transform)
testset = datasets.CIFAR10(root='./data/', train=False, download=True, transform=data_transform)

# exercise C3, C4: workers are set by argument
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.workers)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# select model
# net = ResNet18().to(device)
net = ResNet18()
#net = ResNet18NoBatchNorm()  # model without batch norm

# if device == 'cuda':
if use_gpu:
    print("Using %d GPUs" % gpu_count)
    net = net.cuda()
    device_list = list(range(0, gpu_count))
    if gpu_count > 1:
        net = torch.nn.DataParallel(net, device_ids=device_list)
    # cudnn.benchmark = True  # optimizes?

# Exercise C6: select optimizer and loss function
if args.optim == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
elif args.optim == "nesterov":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
elif args.optim == "adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.decay)
elif args.optim == "adadelta":
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=args.decay)
elif args.optim == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
criterion = torch.nn.CrossEntropyLoss()


# training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    training_time = 0
    parallel_gpu_time = 0  # Q2: time the gpu speedup time including compute time + data IO
    parallel_gpu_timer_start = time.perf_counter()  # begin here so as to include data loading time
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Q1: start recording training time
        t_start = time.perf_counter()
        # inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        # loss = criterion(outputs, targets).to(device)
        loss = criterion(outputs, targets)
        if use_gpu:
            loss = loss.cuda()
        loss.backward()
        optimizer.step()
        # train_loss += loss.item()
        train_loss += loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        correct += (float)(predicted.eq(targets).sum().cpu().detach())

        t_end = time.perf_counter()
        training_time += (t_end - t_start)

        parallel_gpu_timer_end = time.perf_counter()
        parallel_gpu_time += (parallel_gpu_timer_end - parallel_gpu_timer_start)
        parallel_gpu_timer_start = time.perf_counter()  # reset timer for the next batch iteration

        if batch_idx % 10 == 0:
            # change loss.data to loss.data.cpu()[0] when running on pytorch 0.3
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}% ({}/{})'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                train_loss.data.cpu()[0] / (batch_idx + 1), 100.*correct/total, correct, total))


    # after epoch prints
    training_accuracy = 100. * correct / len(train_loader.dataset)  # average accuracy
    print('\ttraining set: Average loss: {:.4f}, Accuracy: {:.2f}% ({}/{})'.format(
        train_loss.data.cpu()[0] / (len(train_loader.dataset) / batch_size), training_accuracy, correct, len(train_loader.dataset)))
    print('\ttraining_time: {:.3f}'.format(
        training_time))
    print('\tparallel_per_gpu_time: {:.3f}'.format(
            parallel_gpu_time))
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total params: " + str(total_params))

if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
