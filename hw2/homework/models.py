import torch.nn


class CNNClassifier(torch.nn.Module):
	# From lecture
	class Block(torch.nn.Module):
		def __init__(self, n_input, n_output, stride=1):
			super().__init__()
			self.net = torch.nn.Sequential(
			  torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
			  torch.nn.ReLU(),
			  torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
			  torch.nn.ReLU()
			)
		
		def forward(self, x):
			return self.net(x)

	def __init__(self):
		"""
		Your code here
		"""
		super().__init__()        

		# From lecture
		L = [
			torch.nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		]
		
		layers = [32, 64] # conv output channels
		c = 32 # current input channels
		kernel_size = 3

		for l in layers:
			L.append(self.Block(c, l, stride=2))
			c = l

		self.network = torch.nn.Sequential(*L)
		self.classifier = torch.nn.Linear(c, 6)
	
	
	def forward(self, x):
		# From lecture
		z = self.network(x)
		z = z.mean(dim=[2,3])
		return self.classifier(z)


def save_model(model):
	from torch import save
	from os import path
	if isinstance(model, CNNClassifier):
		return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cc.th'))
	raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
	from torch import load
	from os import path
	r = CNNClassifier()
	r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
	return r
