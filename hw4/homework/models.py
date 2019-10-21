import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
	H, W = heatmap.shape
	maxpool2d = torch.nn.functional.max_pool2d
	padding = int(max_pool_ks/2)
	max2d = maxpool2d(heatmap[None, None], kernel_size=max_pool_ks, stride=1,
					padding=padding, return_indices=True)
	max_values, max_indices = max2d

	is_local_max = heatmap == max_values.view((H,W))
	is_large_enough = max_values > min_score	
	local_maxes = max_values[is_local_max & is_large_enough]
	local_max_indices = max_indices[is_local_max & is_large_enough]

	topk = torch.topk(local_maxes, k=min(max_det, len(local_maxes)))
	output = []
	for v, i in zip(topk[0], topk[1]):
		idx = local_max_indices[i]
		x, y = int(idx % W), int(idx / W)
		output.append((v.item(), x, y))
		
	return output


class FCN(torch.nn.Module):
	class Block(torch.nn.Module):
		def __init__(self, n_input, n_output, kernel_size=3, stride=2):
			super().__init__()
			self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
									  stride=stride)
			self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
			self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
			self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

		def forward(self, x):
			return F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x)))))) + self.skip(x)
	class UpBlock(torch.nn.Module):
		def __init__(self, n_input, n_output, kernel_size=3, stride=2):
			super().__init__()
			self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
									  stride=stride, output_padding=1)

		def forward(self, x):
			return F.relu(self.c1(x))

	def __init__(self, layers=[16, 32, 64, 96, 128], n_output_channels=5, kernel_size=3, use_skip=True):
		super().__init__()
		self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
		self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

		self.batch_norm = torch.nn.BatchNorm2d(3)

		c = 3
		self.use_skip = use_skip
		self.n_conv = len(layers)
		skip_layer_size = [3] + layers[:-1]
		for i, l in enumerate(layers):
			self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
			c = l
		for i, l in list(enumerate(layers))[::-1]:
			self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
			c = l
			if self.use_skip:
				c += skip_layer_size[i]
		self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

	def forward(self, x):
		z = self.batch_norm(x)
		up_activation = []
		for i in range(self.n_conv):
			# Add all the information required for skip connections
			up_activation.append(z)
			z = self._modules['conv%d'%i](z)

		for i in reversed(range(self.n_conv)):
			z = self._modules['upconv%d'%i](z)
			# Fix the padding
			z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
			# Add the skip connection
			if self.use_skip:
				z = torch.cat([z, up_activation[i]], dim=1)
		return self.classifier(z)


class Detector(torch.nn.Module):
	class Block(torch.nn.Module):
		def __init__(self, n_input, n_output, kernel_size=3, stride=2):
			super().__init__()
			self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
									  stride=stride)
			self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
			self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
			self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

		def forward(self, x):
			return F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x)))))) + self.skip(x)
	class UpBlock(torch.nn.Module):
		def __init__(self, n_input, n_output, kernel_size=3, stride=2):
			super().__init__()
			self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
									  stride=stride, output_padding=1)

		def forward(self, x):
			return F.relu(self.c1(x))

	def __init__(self, layers=[16, 32, 64, 96, 128], n_output_channels=3,
		kernel_size=3, use_skip=True, min_score=[-5, -5, -5], use_FCN=True):
		super().__init__()
		self.min_score = min_score
		self.FCN = FCN
		self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
		self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

		c = 3
		self.use_skip = use_skip
		self.n_conv = len(layers)
		skip_layer_size = [3] + layers[:-1]
		for i, l in enumerate(layers):
			self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
			c = l
		for i, l in list(enumerate(layers))[::-1]:
			self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
			c = l
			if self.use_skip:
				c += skip_layer_size[i]
		self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

		################ use pretrained FCN ################

		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		from os import path

		if use_FCN:
			model = FCN()
			model_path = path.join(path.dirname(path.abspath(__file__)), 'fcn.th')
		else:
			model = Detector()
			model_path = path.join(path.dirname(path.abspath(__file__)), 'det.th')

		model.load_state_dict(torch.load(model_path))
		model.to(device)

		self.device = device
		self.model = model

	def forward(self, x):
		z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
		up_activation = []
		for i in range(self.n_conv):
			# Add all the information required for skip connections
			up_activation.append(z)
			z = self._modules['conv%d'%i](z)

		for i in reversed(range(self.n_conv)):
			z = self._modules['upconv%d'%i](z)
			# Fix the padding
			z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
			# Add the skip connection
			if self.use_skip:
				z = torch.cat([z, up_activation[i]], dim=1)
		return self.classifier(z)

	def detect(self, image):
		"""
		   Your code here.
		   Implement object detection here.
		   @image: 3 x H x W image
		   @return: List of detections [(class_id, score, cx, cy), ...],
					return no more than 100 detections per image
		   Hint: Use extract_peak here
		"""
		

		image = image[None,:,:,:].to(self.device)
		heatmaps = self.model(image)[0]
		heatmaps = heatmaps[[1,3,4],:,:]

		all_detections = []
		for i in range(3):
			detections = []
			heatmap = heatmaps[i]
			peaks = extract_peak(heatmap, min_score=0)
			[detections.append((i, *p)) for p in peaks]
			detections = sorted(detections, key=lambda x: x[1])
			all_detections += detections[-50:]

		all_detections = sorted(all_detections, key=lambda x: x[1])
		return all_detections[::-1][:100]#, heatmaps

	def detect_with_size(self, image):
		"""
		   Your code here. (extra credit)
		   Implement object detection here.
		   @image: 3 x H x W image
		   @return: List of detections [(class_id, score cx, cy, w/2, h/2), ...],
					return no more than 100 detections per image
		   Hint: Use extract_peak here
		"""
		raise NotImplementedError('Detector.detect_with_size')


def save_model(model, suffix=''):
	from torch import save
	from os import path
	return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det{}.th'.format(suffix)))


def load_model():
	from torch import load
	from os import path
	r = Detector()
	r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
	return r


if __name__ == '__main__':
	"""
	Shows detections of your detector
	"""
	from .utils import DetectionSuperTuxDataset
	dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
	import torchvision.transforms.functional as TF
	from pylab import show, subplots
	import matplotlib.patches as patches

	fig, axs = subplots(3, 4)
	model = load_model()
	for i, ax in enumerate(axs.flat):
		im, kart, bomb, pickup = dataset[i]
		ax.imshow(TF.to_pil_image(im), interpolation=None)
		for k in kart:
			ax.add_patch(
				patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
		for k in bomb:
			ax.add_patch(
				patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
		for k in pickup:
			ax.add_patch(
				patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
		for c, s, cx, cy in model.detect(im):
			ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
		ax.axis('off')
	show()
