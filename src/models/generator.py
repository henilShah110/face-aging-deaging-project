import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
#input and output shape remains same
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.InstanceNorm2d(channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # These layers learn the difference
            # Here : Wrinkles, Texture
            # Then just output this characteristics
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x) #SKIP CONNECTION
    
class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.initial = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=1,
                padding=3
            ), #output shape : [4,64, 128, 128] : batch_size,channels,h,w

            nn.InstanceNorm2d(64),

            nn.ReLU(inplace=True)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ), #output : [4, 128, 64, 64]

            nn.InstanceNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1
            ), # output : [4, 256, 32, 32]

            nn.InstanceNorm2d(256),

            nn.ReLU(inplace=True)
        )

        self.residuals = nn.Sequential(

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.upsample = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.InstanceNorm2d(128),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.InstanceNorm2d(64),

            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(

            nn.Conv2d(
                64,
                3,
                kernel_size=7,
                stride=1,
                padding=3
            ),

            nn.Tanh() # tanh is to keep the outputs between 1 to -1
        )

    def forward(self, x):
        return self.final(
            self.upsample(
                self.residuals(
                    self.downsample(
                        self.initial(x)
                    )
                )
            )
        )    