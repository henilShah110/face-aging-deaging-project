import torch
import torch.nn as nn

adversarial_loss = nn.MSELoss() 
# large loss larger punishment, small loss smaller punishment
# punishments are relative

cycle_loss = nn.L1Loss()
# sum of all all difference across pixels

identity_loss = nn.L1Loss()