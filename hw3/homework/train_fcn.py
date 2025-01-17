import torch
import numpy as np
from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
from torchvision import transforms
import torch.nn.functional as F
import pickle


def augment(image, label):
        transform = dense_transforms.Compose([
            dense_transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
            dense_transforms.RandomHorizontalFlip(),
            dense_transforms.ToTensor(),
            dense_transforms.Normalize(np.zeros(3), np.ones(3)),
        ])
        return transform(image, label)


def no_augment(image, label):
    transform = dense_transforms.Compose([
        dense_transforms.ToTensor()
    ])
    return transform(image, label)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


class ClassificationLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        weight = torch.FloatTensor([1/x for x in DENSE_CLASS_DISTRIBUTION])
        self.weight = weight.to(device)

    def forward(self, input, target):        
        return F.cross_entropy(input, target, weight=self.weight)


def train(args):
    from os import path
    #model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = FCN().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))
        global_step = pickle.load(open('global_step.p', 'rb'))
    else:
        global_step = 0

    optimizer = torch.optim.Adam(model.parameters())#, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    loss      = ClassificationLoss(device)

    train_data = load_dense_data('dense_data/train', transform=augment)
    valid_data = load_dense_data('dense_data/valid', transform=no_augment)
    
    for epoch in range(args.num_epoch):
        model.train()
        acc_vals = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            logit      = model(img)

            shape = label.shape
            label = torch.squeeze(label.long())

            loss_val = loss(logit, label)
            acc_val  = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            acc_vals.append(acc_val.detach().cpu().numpy())

            if global_step % 40 == 0:
                print('{}: loss = {}'.format(global_step, loss_val))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        
        avg_acc = sum(acc_vals) / len(acc_vals)
        scheduler.step(np.mean(acc_vals))
        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        avg_vacc = sum(acc_vals) / len(acc_vals)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)
        
        print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        save_model(model)
        pickle.dump(global_step, open('global_step.p', 'wb'))
    
    pickle.dump(global_step, open('global_step.p', 'wb'))
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
