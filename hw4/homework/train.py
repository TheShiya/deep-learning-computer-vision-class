import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb

DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]

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
    w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
    loss = torch.nn.BCEWithLogitsLoss(weight=w / w.mean()).to(device)

    import inspect
    #transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    transform = dense_transforms.Compose([
        dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap(),
        ])
    
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4)

    for epoch in range(args.num_epoch):
        model.train()
        conf = ConfusionMatrix()
        for img, label in train_data:
            img, label = img.to(device), label.to(device).long()

            logit = model(img)
            loss_val = loss(logit, label)
            if train_logger is not None and global_step % 100 == 0:
                train_logger.add_image('image', img[0], global_step)
                train_logger.add_image('label', np.array(dense_transforms.label_to_pil_image(label[0].cpu()).
                                                         convert('RGB')), global_step, dataformats='HWC')
                train_logger.add_image('prediction', np.array(dense_transforms.
                                                              label_to_pil_image(logit[0].argmax(dim=0).cpu()).
                                                              convert('RGB')), global_step, dataformats='HWC')

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            if global_step % 20 == 0:
                print('{}: loss: {}'.format(global_step, loss_val))

            conf.add(logit.argmax(1), label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if train_logger:
            train_logger.add_scalar('global_accuracy', conf.global_accuracy, global_step)
            train_logger.add_scalar('average_accuracy', conf.average_accuracy, global_step)
            train_logger.add_scalar('iou', conf.iou, global_step)

        model.eval()
        val_conf = ConfusionMatrix()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device).long()
            logit = model(img)
            val_conf.add(logit.argmax(1), label)

        if valid_logger is not None:
            valid_logger.add_image('image', img[0], global_step)
            valid_logger.add_image('label', np.array(dense_transforms.label_to_pil_image(label[0].cpu()).
                                                     convert('RGB')), global_step, dataformats='HWC')
            valid_logger.add_image('prediction', np.array(dense_transforms.
                                                          label_to_pil_image(logit[0].argmax(dim=0).cpu()).
                                                          convert('RGB')), global_step, dataformats='HWC')

        if valid_logger:
            valid_logger.add_scalar('global_accuracy', val_conf.global_accuracy, global_step)
            valid_logger.add_scalar('average_accuracy', val_conf.average_accuracy, global_step)
            valid_logger.add_scalar('iou', val_conf.iou, global_step)


        if True or valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f \t iou = %0.3f \t val iou = %0.3f' %
                  (epoch, conf.global_accuracy, val_conf.global_accuracy, conf.iou, val_conf.iou))
        
        pickle.dump(global_step, open('global_step.p', 'wb'))
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
