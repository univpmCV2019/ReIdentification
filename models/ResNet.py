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
		self.classifier = nn.Linear(self.feat_dim/4, num_classes)
		self.bilinear=nn.Bilinear(2048/4,2048/4,2048/4)
		

	def forward(self, x, z):
		b = x.size(0)
		t = x.size(1)
		bd = z.size(0)
		td = z.size(1)
		
		
		
		#Rete base RGB 
		x = x.view(b*t,x.size(2), x.size(3), x.size(4)) 
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:]) #avg pool non ha return_indices
		x = x.view(b,t,-1)
		x = x.permute(0,2,1)
		f = F.avg_pool1d(x,t)
		f = f.view(b*4, self.feat_dim/4)
		#print(f.size())
		
		#Rete Depth 
		z = z.view(bd*td,z.size(2), z.size(3), z.size(4))
		z = self.base(z)
		z = F.avg_pool2d(z, z.size()[2:]) #avg pool non ha return_indices
		z = z.view(bd,td,-1)
		z = z.permute(0,2,1)
		fd = F.avg_pool1d(z,td)
		fd = fd.view(bd*4, self.feat_dim/4) 
		
		
		
		if not self.training:
			return f #rivedere questo 
		y = self.bilinear(f,fd) #Uniamo  
		#riaggiustiamo dimensioni
		f = f.view(6, -1)
		fd = fd.view(6, -1)
		y = y.view(6,-1)

		if self.loss == {'xent'}:
			return y
		elif self.loss == {'xent', 'htri'}:
			return y, f, fd
		elif self.loss == {'cent'}:
			return y, f, fd 
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
		self.classifier = nn.Linear(self.feat_dim/4, num_classes)
		self.bilinear=nn.Bilinear(2048/4,2048/4,2048/4)
		self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
		self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
	def forward(self, x, z):
		b = x.size(0)
		t = x.size(1)
		bd = z.size(0)
		td = z.size(1)
		#RGB
		x = x.view(b*t, x.size(2), x.size(3), x.size(4))
		x = self.base(x)
		a = F.relu(self.attention_conv(x))
		a = a.view(b, t, self.middle_dim)
		a = a.permute(0,2,1)
		a = F.relu(self.attention_tconv(a))
		a = a.view(b, t)
		x = F.avg_pool2d(x, x.size()[2:])
		#Depth
		z = z.view(bd*td, z.size(2), z.size(3), z.size(4))
		z = self.base(z)
		ad = F.relu(self.attention_conv(z))
		ad = ad.view(bd, td, self.middle_dim)
		ad = ad.permute(0,2,1)
		ad = F.relu(self.attention_tconv(ad))
		ad = ad.view(bd, td)
		z = F.avg_pool2d(z, z.size()[2:])
		
		
		if self. att_gen=='softmax':
			a = F.softmax(a, dim=1)
			ad = F.softmax(ad, dim=1)
		elif self.att_gen=='sigmoid':
			a = F.sigmoid(a)
			a = F.normalize(a, p=1, dim=1)
			ad = F.sigmoid(ad)
			ad = F.normalize(ad, p=1, dim=1)
		else: 
			raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
		x = x.view(b, t, -1)
		a = torch.unsqueeze(a, -1)
		a = a.expand(b, t, self.feat_dim)
		att_x = torch.mul(x,a)
		att_x = torch.sum(att_x,1)
		
		z = z.view(bd, td, -1)
		ad = torch.unsqueeze(ad, -1)
		ad = ad.expand(bd, td, self.feat_dim/4)
		att_z = torch.mul(z,ad)
		att_z = torch.sum(att_z,1)
		
		f = att_x.view(b,self.feat_dim/4)
		fd = att_z.view(bd,self.feat_dim/4)
		
		if not self.training:
			return f
		y = self.bilinear(f,fd) #Uniamo  
		#riaggiustiamo dimensioni
		f = f.view(6, -1)
		fd = fd.view(6, -1)
		y = y.view(6,-1)

		if self.loss == {'xent'}:
			return y
		elif self.loss == {'xent', 'htri'}:
			return y, f, fd
		elif self.loss == {'cent'}:
			return y, f, fd
		else:
			raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50RNN(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet50rnn, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])
		self.hidden_dim = 512
		self.feat_dim = 2048
		self.classifier = nn.Linear(self.hidden_dim, num_classes)
		self.bilinear=nn.Bilinear(2048/4,2048/4,2048/4)
		self.lstm = nn.LSTM(input_size=self.feat_dim/4, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
	def forward(self, x, z):
		b = x.size(0)
		t = x.size(1)
		bd = z.size(0)
		bd = z.size(1)
		#RGB
		x = x.view(b*t,x.size(2), x.size(3), x.size(4))
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:])
		x = x.view(b,t,-1)
		output, (h_n, c_n) = self.lstm(x)
		output = output.permute(0, 2, 1)
		f = F.avg_pool1d(output, t)
		f = f.view(b, self.hidden_dim)
		#Depth
		z = z.view(bd*td,z.size(2), z.size(3), z.size(4))
		z = self.base(z)
		z = F.avg_pool2d(z, z.size()[2:])
		z = z.view(bd,td,-1)
		outputd, (h_n, c_n) = self.lstm(z)
		outputd = outputd.permute(0, 2, 1)
		fd = F.avg_pool1d(outputd, td)
		fd = fd.view(bd, self.hidden_dim)
		if not self.training:
			return f
			
		y = self.bilinear(f,fd) #Uniamo  
		#riaggiustiamo dimensioni
		f = f.view(6, -1)
		fd = fd.view(6, -1)
		y = y.view(6,-1)

		if self.loss == {'xent'}:
			return y
		elif self.loss == {'xent', 'htri'}:
			return y, f, fd
		elif self.loss == {'cent'}:
			return y, f, fd
		else:
			raise KeyError("Unsupported loss: {}".format(self.loss))

