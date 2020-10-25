import torch
import torch.nn as nn

VGG_type = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ],
}

# class Encoder(nn.Module):
#     def __init__(self, in_channels):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1
#         )  # what this will do is halve the h, w of the image
#         self.conv2 = nn.Conv2d(
#             in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
#         )
#         self.bn1 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(
#             in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
#         )
#         self.bn2 = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(
#             in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
#         )
#         self.bn3 = nn.BatchNorm2d(512)
#         self.conv5 = nn.Conv2d(
#             in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=0
#         )
#         self.relu = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.bn1(x)
#         x = self.relu(self.conv3(x))
#         x = self.bn2(x)
#         x = self.relu(self.conv4(x))
#         x = self.bn3(x)
#         x = self.relu(self.conv5(x))

#         return x


class ConvAutoEncoder(nn.Module):
    def __init__(self, channels):
        super(ConvAutoEncoder, self).__init__()
        self.enc = VGGEncoder("VGG16")
        self.dec = Decoder(channels)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)

        return x


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=512,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.upconv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.upconv4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.upconv5 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2)
        self.op = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.upconv1(x))
        x = self.bn1(x)
        x = self.relu(self.upconv2(x))
        x = self.bn2(x)
        x = self.relu(self.upconv3(x))
        x = self.bn3(x)
        x = self.relu(self.upconv4(x))
        x = self.bn4(x)
        x = self.relu(self.upconv5(x))
        output = self.op(x)

        return x


class VGGEncoder(nn.Module):
    def __init__(self, vgg_version, in_channels=3):
        super(VGGEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv(VGG_type[vgg_version])
        # after completing all the conv layer the final matrix will be [ bs , 512, 7 , 7]

    def forward(self, x):
        x = self.conv_layers(x)
        print(x)

        return x

    def create_conv(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x

            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)


# if __name__ == "__main__":
#     sample = torch.randn(1, 3, 64, 64)
#     autoencoder = VGGEncoder("VGG16")
#     print(autoencoder(sample).shape)