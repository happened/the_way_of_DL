import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision

train_batch_size = 60
test_batch_size = 1000
img_size = 28

temp_num = 0


def get_dataloader(train=True):
    assert isinstance(train, bool), "train must be boolean"

    dataset = torchvision.datasets.MNIST('/home/data', train=train, download=True,
                                         transform=torchvision.transforms.Compose(
                                             [torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,)), ]
                                         ))

    batch_size = train_batch_size if train else test_batch_size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


mnist_net = MnistNet()
optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)
train_loss_list = []
train_count_list = []


def train(epoch):
    mode = True
    mnist_net.train(mode=mode)
    train_dataloader = get_dataloader(train=mode)
    print(len(train_dataloader.dataset))
    print(len(train_dataloader))
    for idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_dataloader.dataset),
                       100. * idx / len(train_dataloader), loss.item()))
            train_loss_list.append(loss.item())
            train_count_list.append(idx * train_batch_size + (epoch - 1) * len(train_dataloader))
            torch.save(mnist_net.state_dict(), "./model/mnist_net.pkl")
            torch.save(optimizer.state_dict(), './results/mnist_optimizer.pkl')
            torch.save(mnist_net, "./model/mnist.pt")


def test():
    test_loss = 0
    correct = 0
    mnist_net.eval()
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for data, target in test_dataloader:
            global temp_num
            if temp_num == 0:
                temp_num += 1
            output = mnist_net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)
    ))


if __name__ == "__main__":
    test()
    for i in range(10):
        train(i)
        test()
