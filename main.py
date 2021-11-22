import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

from data import train_transforms, val_transforms
from model import KNOWN_MODELS, make_model


def train(model, device, train_loader, optimizer, epoch):
    model.to(device)
    model.train()
    it = tqdm(train_loader, desc=f"Epoch {epoch}")
    for data, target in it:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        it.set_postfix_str(f"loss={loss.item():.03f}")
        optimizer.step()


def validation(model, device, val_loader, scheduler):
    model.to(device)
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(validation_loss)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch-size', type=int, default=4, metavar='B',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    parser.add_argument('--model-name', type=str, default='efficientnet', metavar='MN',
                        help=f'model name; one of {", ".join(KNOWN_MODELS.keys())} (default: efficientnet).')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='when enabled, prints the model after building it, then exit (default: False).')
    parser.add_argument('--freeze-weights', action='store_true', default=False,
                        help='when enabled, freezes the hidden weights and only trains the classifier head, as opposed to fine-tuning the entire model (default: False).')

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Data initialization and loading

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                             transform=train_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=val_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Neural network, optimizer and scheduler
    print(f'Using {device}')
    model = make_model(args.model_name, finetuning=not args.freeze_weights)

    if args.dry_run:
        print(f"Model: {model}")
        print("Dry run, exiting.")
        exit()

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=0.5, patience=2)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        validation(model, device, val_loader, scheduler)
        model_file = os.path.join(args.experiment,
                                  args.model_name + "_" + str(epoch) + '.pth')
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '.\n')
