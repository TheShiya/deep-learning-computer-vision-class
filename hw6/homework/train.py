from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    import pickle
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))
        global_step = pickle.load(open('global_step.p', 'rb'))
    else:
        global_step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.MSELoss(reduction='mean')

    transform = dense_transforms.Compose([
        dense_transforms.ColorJitter(0.7, 0.7, 0.7, 0.2),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor(),
        ])    
    train_data = load_data('drive_data', transform=transform)
    valid_data = load_data('drive_data', transform=dense_transforms.ToTensor())

    batch_size = 128
    x_center = torch.FloatTensor([128//2]*batch_size).to(device)

    model = model.to(device)
    for epoch in range(args.num_epoch):
        model.train()
        train_losses = []        
        for img, label in train_data:
            img, label = img.to(device).float(), label.to(device).float()
            logit = model(img).float()
            # x_label, y_label = label[:,0], label[:,1]
            # x_logit, y_logit = logit[:,0], logit[:,1]
            loss_val = loss(logit, label)#.pow(1.5)
            train_losses.append(loss_val)

            if epoch > 0 and train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            if global_step % 10 == 0:
                print('{}: loss: {}'.format(global_step, loss_val))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        valid_losses = []
        count = 0
        # for img, label in valid_data:
        #     img, label = img.to(device).float(), label.to(device).float()
        #     logit = model(img).float()
        #     valid_loss = loss(logit, label)
        #     valid_losses.append(valid_loss)
        #     count += 0
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        #avg_valid_loss = sum(valid_losses) / len(valid_losses)

        if valid_logger is None or train_logger is None:
            train_logger.add_scalar('avg_loss', avg_train_loss, epoch)
            #valid_logger.add_scalar('avg_loss', avg_valid_loss, epoch)

        print('epoch %-3d \t train = %0.3f \t valid =' % (epoch, avg_train_loss))
        
        pickle.dump(global_step, open('global_step.p', 'wb'))
        save_model(model, suffix=str(epoch))
    save_model(model, suffix='')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
