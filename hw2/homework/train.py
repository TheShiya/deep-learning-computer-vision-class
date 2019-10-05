from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F


# From HW1
class ClassificationLoss(torch.nn.Module):
	def forward(self, input, target):
		return F.cross_entropy(input, target)



def train(args):
	print('started')

	from os import path
	model = CNNClassifier()
	print('model')
	train_logger, valid_logger = None, None
	if args.log_dir is not None:
		train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
		valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

	"""
	Your code here, modify your HW1 code
	
	"""

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	learning_rate     = args.learning_rate
	num_epoch         = args.num_epoch
	continue_training = args.continue_training

	model.to(device)
	if continue_training:
		from os import path
		model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cc.th')))

	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	loss = ClassificationLoss()

	train_data = load_data('data/train')
	valid_data = load_data('data/valid')

	for epoch in range(num_epoch):
		model.train()
		loss_vals, acc_vals, vacc_vals = [], [], []
		
		print('Starting epoch', epoch)
		i = 0
		for img, label in train_data:
			img, label = img.to(device), label.to(device)

			logit = model(img)
			loss_val = loss(logit, label)
			acc_val = accuracy(logit, label)

			loss_vals.append(loss_val.detach().cpu().numpy())
			acc_vals.append(acc_val.detach().cpu().numpy())

			optimizer.zero_grad()
			loss_val.backward()
			optimizer.step()

			i += 1
			train_logger.add_scalar('train/loss', loss_val, epoch+len(train_data) + i)
			if i % 20 == 0:
				print('{}: loss={}'.format(i, loss_val))

		avg_loss = sum(loss_vals) / len(loss_vals)
		avg_acc = sum(acc_vals) / len(acc_vals)

		model.eval()
		for img, label in valid_data:
			img, label = img.to(device), label.to(device)
			vacc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
		avg_vacc = sum(vacc_vals) / len(vacc_vals)

		train_logger.add_scalar('train/accuracy', avg_acc, epoch)
		valid_logger.add_scalar('train/valid', avg_vacc, epoch)

		print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))
	save_model(model)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', default='log')
	# Put custom arguments here

	parser.add_argument('-n', '--num_epoch', type=int, default=5)
	parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
	parser.add_argument('-c', '--continue_training', action='store_true')

	args = parser.parse_args()
	train(args)
