from __future__ import absolute_import

import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['ResNet50TP', 'ResNet50TA', 'ResNet50RNN']


class ResNet50TP(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet50TP, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])
		self.feat_dim = 2048
		self.classifier = nn.Linear(self.feat_dim, num_classes)
		'''
		# Load pre-trained VGG-16 weights to two separate variables.
		# They will be used in defining the depth and RGB encoder sequential layers.
		feats = list(models.vgg16(pretrained=True).features.children())
		feats2 = list(models.vgg16(pretrained=True).features.children())
		
		# Average the first layer of feats variable, the input-layer weights of VGG-16,
		# over the channel dimension, as depth encoder will be accepting one-dimensional
		# inputs instead of three.
		avg = torch.mean(feats[0].cuda(gpu_device).weight.data, dim=1)
		avg = avg.unsqueeze(1)
		
		# DEPTH ENCODER
		self.conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1).cuda(gpu_device)
		self.conv11d.weight.data = avg

		self.CBR1_D = nn.Sequential(
			nn.BatchNorm2d(64).cuda(gpu_device),
			feats[1].cuda(gpu_device),
			feats[2].cuda(gpu_device),
			nn.BatchNorm2d(64).cuda(gpu_device),
			feats[3].cuda(gpu_device),
		)
		self.CBR2_D = nn.Sequential(
			feats[5].cuda(gpu_device),
			nn.BatchNorm2d(128).cuda(gpu_device),
			feats[6].cuda(gpu_device),
			feats[7].cuda(gpu_device),
			nn.BatchNorm2d(128).cuda(gpu_device),
			feats[8].cuda(gpu_device),
		)
		self.CBR3_D = nn.Sequential(
			feats[10].cuda(gpu_device),
			nn.BatchNorm2d(256).cuda(gpu_device),
			feats[11].cuda(gpu_device),
			feats[12].cuda(gpu_device),
			nn.BatchNorm2d(256).cuda(gpu_device),
			feats[13].cuda(gpu_device),
			feats[14].cuda(gpu_device),
			nn.BatchNorm2d(256).cuda(gpu_device),
			feats[15].cuda(gpu_device),
		)
		self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)

		self.CBR4_D = nn.Sequential(
			feats[17].cuda(gpu_device),
			nn.BatchNorm2d(512).cuda(gpu_device),
			feats[18].cuda(gpu_device),
			feats[19].cuda(gpu_device),
			nn.BatchNorm2d(512).cuda(gpu_device),
			feats[20].cuda(gpu_device),
			feats[21].cuda(gpu_device),
			nn.BatchNorm2d(512).cuda(gpu_device),
			feats[22].cuda(gpu_device),
		)
		self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

		self.CBR5_D = nn.Sequential(
			feats[24].cuda(gpu_device),
			nn.BatchNorm2d(512).cuda(gpu_device),
			feats[25].cuda(gpu_device),
			feats[26].cuda(gpu_device),
			nn.BatchNorm2d(512).cuda(gpu_device),
			feats[27].cuda(gpu_device),
			feats[28].cuda(gpu_device),
			nn.BatchNorm2d(512).cuda(gpu_device),
			feats[29].cuda(gpu_device),
		)
		
		# RGB DECODER
		self.CBR5_Dec = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Dropout(p=0.5).cuda(gpu_device),
		)

		self.CBR4_Dec = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(512, 256, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Dropout(p=0.5).cuda(gpu_device),
		)

		self.CBR3_Dec = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(256,	128, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Dropout(p=0.5).cuda(gpu_device),
		)

		self.CBR2_Dec = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
		)

		self.CBR1_Dec = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(64, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
		)

		'''

	def forward(self, x, z):
		b = x.size(0)
		t = x.size(1)
		bd = z.size(0)
		td = z.size(1)
		print(b)
		print(t)
		print(bd)
		print(td)
		sys.exit("Fermata esecuzione qui")
		'''
		# DEPTH ENCODER
		# Stage 1
		d = self.conv11d(z)
		d_1 = self.CBR1_D(d)
		d, id1_d = F.max_pool2d(d_1, kernel_size=2, stride=2, return_indices=True)

		# Stage 2
		d_2 = self.CBR2_D(d)
		d, id2_d = F.max_pool2d(d_2, kernel_size=2, stride=2, return_indices=True)

		# Stage 3
		d_3 = self.CBR3_D(d)
		d, id3_d = F.max_pool2d(d_3, kernel_size=2, stride=2, return_indices=True)
		d = self.dropout3_d(d)

		# Stage 4
		d_4 = self.CBR4_D(d)
		d, id4_d = F.max_pool2d(d_4, kernel_size=2, stride=2, return_indices=True)
		d = self.dropout4_d(d)

		# Stage 5
		d_5 = self.CBR5_D(d)
		'''
		#Rete base Depth
		
		z = z.view(bd*td,z.size(2), z.size(3), z.size(4))
		z_1 = self.base(z)
		z = F.avg_pool2d(z_1, z_1.size()[2:])
		z = z.view(bd,td,-1)
		z = z.permute(0,2,1)
		fd_1 = F.avg_pool1d(z,td)
		fd = fd_1.view(bd, self.feat_dim)
		
		#Rete base RGB 
		
		x = x.view(b*t,x.size(2), x.size(3), x.size(4))
		x = self.base(x)
		x = torch.add(x,z_1)
		x = F.avg_pool2d(x, x.size()[2:]) #avg pool non ha return_indices
		x = x.view(b,t,-1)
		x = x.permute(0,2,1)
		x = torch.add(x,fd_1)
		f = F.avg_pool1d(x,t)
		f = f.view(b, self.feat_dim)
		'''
		
		# RGB ENCODER
		# Stage 1
		y = self.CBR1_RGB(rgb_inputs)
		y = torch.add(y, x_1)
		y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

		# Stage 2
		y = self.CBR2_RGB(y)
		y = torch.add(y, x_2)
		y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

		# Stage 3
		y = self.CBR3_RGB(y)
		y = torch.add(y, x_3)
		y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout3(y)

		# Stage 4
		y = self.CBR4_RGB(y)
		y = torch.add(y,x_4)
		y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout4(y)

		# Stage 5
		y = self.CBR5_RGB(y)
		y = torch.add(y, x_5)
		y_size = y.size()

		y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout5(y)

		if self.use_class:
			# FC Block for Scene Classification
			y_class = y.view(y.size(0), -1)
			y_class = self.ClassHead(y_class)
			
		'''

		'''
		# DECODER
		# Stage 5 dec
		y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
		y = self.CBR5_Dec(y)

		# Stage 4 dec
		y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
		y = self.CBR4_Dec(y)

		# Stage 3 dec
		y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
		y = self.CBR3_Dec(y)

		# Stage 2 dec
		y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
		y = self.CBR2_Dec(y)

		# Stage 1 dec
		y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
		y = self.CBR1_Dec(y)
		'''
		'''
		#DECODER
		#con avg_pool2d non funziona ma dovrebbe essere qualcosa di simile
		# Stage 2 dec
		y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
		y = self.CBR2_Dec(y)

		# Stage 1 dec
		y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
		y = self.CBR1_Dec(y)
		'''
		
		if not self.training:
			return f
		y = self.classifier(f)

		if self.loss == {'xent'}:
			return y
		elif self.loss == {'xent', 'htri'}:
			return y, f
		elif self.loss == {'cent'}:
			return y, f
		else:
			raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet50TA, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])
		self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
		self.feat_dim = 2048 # feature dimension
		self.middle_dim = 256 # middle layer dimension
		self.classifier = nn.Linear(self.feat_dim, num_classes)
		self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
		self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
	def forward(self, x):
		b = x.size(0)
		t = x.size(1)
		x = x.view(b*t, x.size(2), x.size(3), x.size(4))
		x = self.base(x)
		a = F.relu(self.attention_conv(x))
		a = a.view(b, t, self.middle_dim)
		a = a.permute(0,2,1)
		a = F.relu(self.attention_tconv(a))
		a = a.view(b, t)
		x = F.avg_pool2d(x, x.size()[2:])
		if self. att_gen=='softmax':
			a = F.softmax(a, dim=1)
		elif self.att_gen=='sigmoid':
			a = F.sigmoid(a)
			a = F.normalize(a, p=1, dim=1)
		else: 
			raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
		x = x.view(b, t, -1)
		a = torch.unsqueeze(a, -1)
		a = a.expand(b, t, self.feat_dim)
		att_x = torch.mul(x,a)
		att_x = torch.sum(att_x,1)
		
		f = att_x.view(b,self.feat_dim)
		if not self.training:
			return f
		y = self.classifier(f)

		if self.loss == {'xent'}:
			return y
		elif self.loss == {'xent', 'htri'}:
			return y, f
		elif self.loss == {'cent'}:
			return y, f
		else:
			raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50RNN(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet50r, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])
		self.hidden_dim = 512
		self.feat_dim = 2048
		self.classifier = nn.Linear(self.hidden_dim, num_classes)
		self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
	def forward(self, x):
		b = x.size(0)
		t = x.size(1)
		x = x.view(b*t,x.size(2), x.size(3), x.size(4))
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:])
		x = x.view(b,t,-1)
		output, (h_n, c_n) = self.lstm(x)
		output = output.permute(0, 2, 1)
		f = F.avg_pool1d(output, t)
		f = f.view(b, self.hidden_dim)
		if not self.training:
			return f
		y = self.classifier(f)

		if self.loss == {'xent'}:
			return y
		elif self.loss == {'xent', 'htri'}:
			return y, f
		elif self.loss == {'cent'}:
			return y, f
		else:
			raise KeyError("Unsupported loss: {}".format(self.loss))

