import torch
import torch.nn as nn

print("##### CONV_1D #####")
x = torch.rand(10, 20, 4)  # Batch, Sequence (Channels), Features
print('input_size:', x.shape)

conv1d = nn.Conv1d(in_channels=20, out_channels=32, kernel_size=4)
print('kernel_weight_size:', conv1d.weight.shape)
print('kernel_bias_size:', conv1d.bias.shape)

out = conv1d(x)
print('output_size:', out.shape)

#################################################################################
print("\n##### CONV_2D #####")
x = torch.rand(10, 20, 80, 120)  # Batch, Channels, Height, Width
print('input_size:', x.shape)

conv2d = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(2, 2))
print('kernel_weight_size:', conv2d.weight.shape)
print('kernel_bias_size:', conv2d.bias.shape)

out = conv2d(x)
print('output_size:', out.shape)

#################################################################################
print("\n##### CONV_3D #####")
x = torch.rand(10, 20, 50, 80, 120)  # Batch, Channels, Depth, Height, Width
print('input_size:', x.shape)

conv3d = nn.Conv3d(in_channels=20, out_channels=32, kernel_size=(2, 2, 2))
print('kernel_weight_size:', conv3d.weight.shape)
print('kernel_bias_size:', conv3d.bias.shape)

out = conv3d(x)
print('output_size:', out.shape)