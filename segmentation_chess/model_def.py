import torch
import torch.nn as nn
import timm


class FirstFeature(nn.Module):
	def __init__(self, in_channels: int):
		super().__init__()
		self.conv2d = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.conv2d(x)


class FinalOutput(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.conv2d = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.conv2d(x)


class Decoder(nn.Module):
	def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
		super().__init__()
		self.conv2d = nn.Sequential(
			nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
		)
		self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x_up = self.up_sample(x)
		x_cat = torch.cat([x_up, skip], dim=1)
		return self.conv2d(x_cat)


class ReUNet(nn.Module):
	def __init__(self, n_channels: int, n_classes: int):
		super().__init__()
		self.first_feature = FirstFeature(n_channels)
		self.resnet = timm.create_model('resnet101', pretrained=True, features_only=True)
		# ResNet101 feature maps: (64, 256, 512, 1024, 2048)
		self.decoder1 = Decoder(2048, 1024, 1024)
		self.decoder2 = Decoder(1024, 512, 512)
		self.decoder3 = Decoder(512, 256, 256)
		self.decoder4 = Decoder(256, 64, 64)
		self.decoder5 = Decoder(64, n_channels, 32)
		self.final_output = FinalOutput(32, n_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x1 = self.first_feature(x)
		# Encoder features
		E1, E2, E3, E4, E5 = self.resnet(x1)
		# Decoder path
		D1 = self.decoder1(E5, E4)
		D2 = self.decoder2(D1, E3)
		D3 = self.decoder3(D2, E2)
		D4 = self.decoder4(D3, E1)
		D5 = self.decoder5(D4, x)
		return self.final_output(D5)



