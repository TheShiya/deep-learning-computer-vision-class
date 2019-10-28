import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot


def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    model = TCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """
    raise NotImplementedError('train')
    save_model(model)
    
    
def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    
    model = TCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th')))
        global_step = pickle.load(open('global_step.p', 'rb'))
    else:
        global_step = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.BCEWithLogitsLoss().to(device)
    
    train_data = SpeechDataset('data/train.txt')
    valid_data = SpeechDataset('data/valid.txt')

    model = model.to(device)
    for epoch in range(args.num_epoch):
        model.train()
        train_losses = []        
        for img, label, square in train_data:
            img, label = img.to(device).float(), label.to(device).float()
            logit = model(img).float()
            
            print('logit shape::::::::::', logit.shape)
            print('label shape::::::::::', label.shape)
            
            loss_val = loss(logit, label)
            train_losses.append(loss_val)

            if epoch > 0 and train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            if global_step % 100 == 0:
                print('{}: loss: {}'.format(global_step, loss_val))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        valid_losses = []
        count = 0
        for img, label, square in valid_data:         
            img, label = img.to(device).float(), label.to(device).float()
            logit = model(img).float()
            valid_loss = loss(logit, label)
            valid_losses.append(valid_loss)
            count += 0          
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        #avg_valid_loss = sum(valid_losses) / len(valid_losses)

        if valid_logger is None or train_logger is None:
            train_logger.add_scalar('avg_loss', avg_train_loss, epoch)
            #valid_logger.add_scalar('avg_loss', avg_valid_loss, epoch)

        print('epoch %-3d \t train = %0.3f \t valid = %0.3f \t' % (epoch, avg_train_loss, avg_train_loss))
        
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
