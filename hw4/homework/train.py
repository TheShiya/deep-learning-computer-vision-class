import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import pickle

def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()

class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0)

    def __init__(self, size=3):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)

# Distribution of karts, bombs/projectiles, and pickup items
DENSE_CLASS_DISTRIBUTION = [0.77357634, 0.11783845, 0.10858521]

def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))
        global_step = pickle.load(open('global_step.p', 'rb'))
    else:
        global_step = 0
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)
    w = (1 - w / w.sum())**args.gamma
    loss = torch.nn.BCEWithLogitsLoss(weight=w).to(device)

    import inspect
    #transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    transform = dense_transforms.Compose([
        dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap(),
        ])

    transform_valid = dense_transforms.Compose([
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap(),
        ])
    
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4, transform=transform_valid)

    model = model.to(device)
    for epoch in range(args.num_epoch):
        model.train()
        train_losses = []        
        for img, label, square in train_data:
            img, label = img.to(device).float(), label.to(device).float()
            logit = model(img)
            loss_val = loss(logit.permute((0,2,3,1)), label.permute((0,2,3,1)))
            train_losses.append(loss_val)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            if global_step % 100 == 0:
                print('{}: loss: {}'.format(global_step, loss_val))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        valid_losses = []
        for img, label, square in valid_data:            
            img, label = img.to(device).float(), label.to(device).float()
            logit = model(img)
            valid_loss = loss(logit.permute((0,2,3,1)), label.permute((0,2,3,1)))
            valid_losses.append(valid_loss)            
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_valid_loss = sum(valid_losses) / len(valid_losses)

        if valid_logger is None or train_logger is None:
            train_logger.add_scalar('avg_loss', avg_train_loss, epoch)
            valid_logger.add_scalar('avg_loss', avg_valid_loss, epoch)

        print('epoch %-3d \t train = %0.3f \t valid = %0.3f \t' % (epoch, avg_train_loss, avg_valid_loss))
        
        pickle.dump(global_step, open('global_step.p', 'wb'))
        save_model(model, suffix=str(epoch))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
