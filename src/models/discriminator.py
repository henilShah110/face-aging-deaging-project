import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.initial = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU( # turns negative number x into x*0.2
                #normal ReLU stops from learning if gradient becomes negative
                0.2,
                inplace=True
            )
        )

        self.model = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=1,
                padding=1
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            ) # output : [1,512,15,15]
        ) #approx its 1,512,14,14 ; this 14,14 is the patch grid



    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return x
    