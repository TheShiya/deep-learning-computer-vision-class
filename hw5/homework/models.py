import torch

from . import utils


class LanguageModel(object):
	def predict_all(self, some_text):
		"""
		Given some_text, predict the likelihoods of the next character for each substring from 0..i
		The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
		including the first empty string (probability of the first character)

		:param some_text: A string containing characters in utils.vocab, may be an empty string!
		:return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
		"""
		raise NotImplementedError('Abstract function LanguageModel.predict_all')

	def predict_next(self, some_text):
		"""
		Given some_text, predict the likelihood of the next character

		:param some_text: A string containing characters in utils.vocab, may be an empty string!
		:return: a Tensor (len(utils.vocab)) of log-probabilities
		"""
		return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
	"""
	Implements a simple Bigram model. You can use this to compare your TCN to.
	The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
	transitions more often. See https://en.wikipedia.org/wiki/Bigram .
	Use this to debug your `language.py` functions.
	"""

	def __init__(self):
		from os import path
		self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))

	def predict_all(self, some_text):
		return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
	"""
	A simple language model that favours adjacent characters.
	The first character is chosen uniformly at random.
	Use this to debug your `language.py` functions.
	"""

	def predict_all(self, some_text):
		prob = 1e-3*torch.ones(len(utils.vocab), len(some_text)+1)
		if len(some_text):
			one_hot = utils.one_hot(some_text)
			prob[-1, 1:] += 0.5*one_hot[0]
			prob[:-1, 1:] += 0.5*one_hot[1:]
			prob[0, 1:] += 0.5*one_hot[-1]
			prob[1:, 1:] += 0.5*one_hot[:-1]
		return (prob/prob.sum(dim=0, keepdim=True)).log()

vocab_size = 28

class TCN(torch.nn.Module, LanguageModel):
	class CausalConv1dBlock(torch.nn.Module):
		def __init__(self, in_channels, out_channels, kernel_size, dilation=1, is_residual=False):
			"""
			Your code here.
			Implement a Causal convolution followed by a non-linearity (e.g. ReLU).
			Optionally, repeat this pattern a few times and add in a residual block
			:param in_channels: Conv1d parameter
			:param out_channels: Conv1d parameter
			:param kernel_size: Conv1d parameter
			:param dilation: Conv1d parameter
			"""
			super().__init__()
			self.batch_norm_in = torch.nn.BatchNorm1d(in_channels)
			self.pad1 = torch.nn.ConstantPad1d((kernel_size+(kernel_size-1)*dilation-1, 0), 0)
			self.c1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=dilation)
			self.pad2 = torch.nn.ConstantPad1d((kernel_size-1, 0), 0)
			self.c2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1)
			self.pad3 = torch.nn.ConstantPad1d((kernel_size-1, 0), 0)
			self.c3 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1)
			self.relu = torch.nn.ReLU()
			self.skip = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
			self.is_residual = is_residual
			self.batch_norm_out = torch.nn.BatchNorm1d(out_channels)

		def forward(self, x):
			z = self.batch_norm_in(x)
			z = self.relu(self.c1(self.pad1(z)))
			z = self.relu(self.c2(self.pad2(z)))
			z = self.relu(self.c3(self.pad3(z)))
			z = z + int(self.is_residual) * self.skip(x)
			z = self.batch_norm_out(z)
			return z

	def __init__(self):
		"""
		Your code here

		Hint: Try to use many layers small (channels <=50) layers instead of a few very large ones
		Hint: The probability of the first character should be a parameter
		use torch.nn.Parameter to explicitly create it.
		"""
		super().__init__()
		# l = torch.nn.Linear(2, 2)
		# net = torch.nn.Sequential(l, l)
		# self.network = net
		kernel_size = 3

		net = []		
		in_ch = 28
		n_layers = 8
		channels = [30,30,40,40]
		is_residual = [0,1] * (len(channels) // 2)
		for ch, res in zip(channels, is_residual):
			net.append(self.CausalConv1dBlock(in_ch, ch, kernel_size=kernel_size, is_residual=res))
			in_ch = ch
		net.append(torch.nn.Conv1d(in_ch, 28, kernel_size=1))
		self.net = torch.nn.Sequential(*net)
		self.prob_first = torch.nn.Parameter(torch.ones(1, 28, 1)/28)
		self.classifier = torch.nn.LogSoftmax(dim=1)
		self.batch_norm = torch.nn.BatchNorm1d(28)
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	def forward(self, x):
		"""
		Your code here
		Return the logit for the next character for prediction for any substring of x

		@x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
		@return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
		"""
		# B, vocab_size, L = x.shape
		# return torch.randn(B, vocab_size, L+1)
		#print('x shape::::::::::::', x.shape)
		
		B, vocab_size, L = x.shape
		prob_firsts = torch.cat([self.prob_first]*B)
		cat = torch.cat([prob_firsts, x], 2)
		o = self.net(cat)
		o = self.classifier(o)
		return o

	def predict_all(self, some_text):
		"""
		Your code here

		@some_text: a string
		@return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
		"""
		#return torch.log(torch.ones(28, len(some_text)+1)/28)
		tensor = utils.one_hot(some_text)[None]
		o = self.forward(tensor)[0]
		return o


def save_model(model):
	from os import path
	return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
	from os import path
	r = TCN()
	r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
	return r
