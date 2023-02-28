import torch
import torch.nn as nn
import torch.nn.functional as f


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, channels=(1, 64, 128, 256, 512, 1024), kernel_size=3):
        super().__init__()
        self.encoding_blocks = nn.ModuleList([ConvBlock(channels[i], channels[i+1], kernel_size=kernel_size) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.encoding_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        
        return features


class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64), kernel_size=3):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.decoding_blocks = nn.ModuleList([ConvBlock(channels[i], channels[i+1], kernel_size=kernel_size) for i in range(len(channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.decoding_blocks[i](x)
        
        return x

class UNet(nn.Module):
    def __init__(self, num_classes=1, kernel_size=3, feature_channels=16):
        super().__init__()
        self.encoder_channels = [feature_channels * 2**i for i in range(5)]
        self.decoder_channels = self.encoder_channels.copy()
        self.encoder_channels.insert(0,1)
        self.decoder_channels.reverse()

        self.encoder = Encoder(self.encoder_channels, kernel_size)
        self.decoder = Decoder(self.decoder_channels, kernel_size)
        self.head = nn.Conv2d(self.decoder_channels[-1], num_classes, 1)

    def forward(self, x):
        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = f.softmax(self.head(out), dim=0)
        return out



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
