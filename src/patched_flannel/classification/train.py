# Import modules
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
from classification import header
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from classification.utils import (
    initialize_model,
    make_weights_for_balanced_classes_customloader,
    plot_classes_preds_single,
    save_checkpoint,
    CLASSES)
from classification.customloader import COVID_Dataset
from torch.utils.tensorboard import SummaryWriter


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Number of classes
num_classes = header.num_classes

# Feature extract
feature_extract = header.feature_extract

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/' + header.test_name)


def main(opts):
    # Save data
    save_dir = opts.checkpoint

    # Model name
    model_name = opts.arch

    input_size = opts.crop_size

    # Initialize the model for this run
    model_ft, _ = initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    train_dataset = COVID_Dataset(
        (opts.crop_size, opts.crop_size), n_channels=3, n_classes=4, mode='train', opts=opts)
    val_dataset = COVID_Dataset(
        (opts.crop_size, opts.crop_size), n_channels=3, n_classes=4, mode='val', opts=opts)

    image_datasets = {'train': train_dataset, 'val': val_dataset}

    # TODO No oversampling
    if header.sampling_option == 'oversampling':
        train_weights = make_weights_for_balanced_classes_customloader(image_datasets['train'].imgs,
                                                                       len(image_datasets['train'].classes))
        train_weights = torch.DoubleTensor(train_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            train_weights, len(train_weights), replacement=True)

        val_weights = make_weights_for_balanced_classes_customloader(image_datasets['val'].imgs,
                                                                     len(image_datasets['val'].classes))
        val_weights = torch.DoubleTensor(val_weights)
        val_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            val_weights, len(val_weights), replacement=True)

        sampler = {'train': train_sampler, 'val': val_sampler}

    else:
        sampler = {'train': None, 'val': None}

    batch_size = {'train': header.train_batch_size,
                  'val': header.val_batch_size}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size[x],
        sampler=sampler[x], num_workers=opts.workers,
        pin_memory=True) for x in ['train', 'val']}

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Set either feature extraction or train all parameters.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=header.lr, weight_decay=0.1)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist_v, hist_t, hist_f1_v, hist_f1_t, epoch_trained, num_epochs = train_model(
        model_ft, dataloaders_dict,
        criterion, optimizer_ft,
        num_epochs=opts.epochs,
        is_inception=(model_name.startswith("inception")),
        opts=opts)
    plot_train(
        hist_t, hist_v,
        hist_f1_t, hist_f1_v, epoch_trained, num_epochs,
        opts.train_plot)


# Define training and validation_model
def train_model(model, dataloaders, criterion, optimizer,
                num_epochs=header.epoch_max, is_inception=False, opts=None):

    print('')
    print('Training for single best model started..')

    since = time.time()

    val_acc_history = []
    train_acc_history = []

    val_f1_history = []
    train_f1_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_avg = 0.0

    count_ES = 0

    # TODO: put resume from checkpoint
    # Load model and optimizer, saved epoch if 'resume' training.
    if opts.resume and os.path.isfile(os.path.join(
            opts.checkpoint_dir, opts.resume+'.checkpoint.pth.tar')):
        checkpoint = torch.load(os.path.join(
            opts.checkpoint_dir, opts.resume+'.checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trained_epoch = checkpoint['epoch']
        best_avg = checkpoint['best_acc']
        print('Previous model saved at %d epoch was loaded and continue training.' %
              trained_epoch)
    else:
        trained_epoch = 0

    print('Model name:', opts.arch)

    for epoch in range(trained_epoch, num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            y_score = []
            y_true = []
            y_pred = []

            i = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                # labels = labels.to(device)
                labels = labels.to(device=device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        # # Add L1 or L2 regularization
                        # l1_regularization = torch.tensor(
                        #     0).to(device, dtype=float)
                        # for param in model.parameters():
                        #     l1_regularization += torch.norm(param, 1)
                        loss = criterion(outputs, labels)  # + \
                        # header.lambda_l1 * l1_regularization

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                prob = torch.sigmoid(outputs)
                prob_np = prob.detach().cpu().numpy()
                labels_np = np.asarray(labels.cpu())
                pred_np = np.asarray(preds.cpu())

                for x in range(len(inputs)):
                    y_sc = prob_np[x][1]
                    y_score.append(y_sc)

                for x in range(len(labels)):
                    y_tr = labels_np[x]
                    y_true.append(y_tr)

                for x in range(len(inputs)):
                    y_pr = pred_np[x]
                    y_pred.append(y_pr)

                if i == 0 and phase == 'val':
                    writer.add_figure(phase + '_predictions vs. actuals',
                                      plot_classes_preds_single(
                                          model, inputs, labels),
                                      global_step=epoch + 1)

                i += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(y_true, y_pred, average='macro')

            if phase == 'val':
                report_dict = classification_report(
                    y_true, y_pred, output_dict=True, target_names=CLASSES)

                covid19_f1 = report_dict['COVID-19']['f1-score']
                pneumonia_virus_f1 = report_dict['pneumonia_virus']['f1-score']
                pneumonia_bacteria_f1 = report_dict['pneumonia_bacteria']['f1-score']
                normal_f1 = report_dict['normal']['f1-score']

                epoch_avg = (normal_f1 + pneumonia_virus_f1 +
                             pneumonia_bacteria_f1 + covid19_f1)/4

                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} avg: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_f1, epoch_avg))
                print(classification_report(
                    y_true, y_pred, target_names=CLASSES))
                # here
            else:
                # save file
                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_f1))

            # Early stopping
            if phase == 'val':
                if epoch_avg > best_avg - 0.05:
                    count_ES = 0
                else:
                    count_ES += 1

                is_best = epoch_avg > best_avg

                if is_best or epoch % opts.checkpoint_saved_n == 0 or epoch == num_epochs-1:
                    print('Current avg score: %f, Best avg score %f, model saved.' % (
                        epoch_avg, best_avg))
                    best_avg = max(best_avg, epoch_avg)
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_checkpoint({
                        'epoch': epoch+1,
                        'best_acc': best_avg,
                        'model_state_dict': best_model_wts,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, epoch, is_best, opts.checkpoint_dir)
                else:
                    print('Model not saved.')

            # Train and validation accuracy
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

            # Train and validation auc
            if phase == 'val':
                val_f1_history.append(epoch_f1)
            if phase == 'train':
                train_f1_history.append(epoch_f1)

            writer.add_scalars('loss', {phase: epoch_loss}, epoch + 1)
            writer.add_scalars('f1', {phase: epoch_f1}, epoch + 1)

        if count_ES == header.patience:
            num_epochs = epoch + 1
            print('Early stopping at.. %d epoch.' % num_epochs)
            break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, train_acc_history, val_f1_history, train_f1_history, trained_epoch, num_epochs


def plot_train(hist_t, hist_v,
               hist_f1_t, hist_f1_v,
               epoch_trained, num_epochs,
               plot_file):
    # Plot loss and accuracy
    vhist = [h.cpu().numpy() for h in hist_v]
    thist = [h.cpu().numpy() for h in hist_t]

    vhist_f1 = [np.asarray(h) for h in hist_f1_v]
    thist_f1 = [np.asarray(h) for h in hist_f1_t]

    plt.subplot(2, 1, 1)
    plt.title("Accuracy vs. Number of Training Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(epoch_trained + 1, num_epochs + 1),
             vhist, label="Validation")
    plt.plot(range(epoch_trained + 1, num_epochs + 1),
             thist, label="Training")
    plt.legend(loc='upper left', frameon=False)

    plt.subplot(2, 1, 2)
    plt.title("F1 vs. Number of Training Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("F1")
    plt.plot(range(epoch_trained + 1, num_epochs + 1),
             vhist_f1, label="Validation")
    plt.plot(range(epoch_trained + 1, num_epochs + 1),
             thist_f1, label="Training")
    plt.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(plot_file)


if __name__ == '__main__':
    main()
