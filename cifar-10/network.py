import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 100,
                        kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(num_features = 100),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 100, out_channels = 100,
                        kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(num_features = 100),
            nn.ReLU(),
        )
        self.avg_pool1 = nn.AvgPool2d((2, 2))
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 100, out_channels = 100, 
                        kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels = 100, out_channels = 100, 
                        kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        self.avg_pool2 = nn.AvgPool2d(kernel_size = (2, 2))
        
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = 100, out_channels = 100,
                        kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels = 100, out_channels = 100, 
                      kernel_size = (3, 3), padding = 1),
            nn.BatchNorm2d(100), 
            nn.ReLU(),
        )
        self.avg_pool3 = nn.AvgPool2d(kernel_size = (2, 2))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(500, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(160, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(50, 10),
            nn.Softmax(dim = -1),
        )
        

    def forward(self, x):
        
        b1_out = self.block1(x)
        b2_out = self.block2(b1_out)
        p1_out = self.avg_pool1(b2_out + b1_out)
        
        b3_out = self.block3(p1_out)
        p2_out = self.avg_pool2(b3_out + p1_out)
        
        b4_out = self.block4(p2_out)
        p3_out = self.avg_pool3(b4_out + p2_out)
        
        output = self.fc(p3_out)
        return output


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
