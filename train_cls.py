import time
import torch
from progress.bar import Bar
import numpy as np


def train_step_cls(train_loader, model, epoch, optimizer, criterion, args, global_parameters):
    # switch to train mode
    model.train()
    model.load_state_dict(global_parameters, strict=True)
    epoch_loss = 0.0
    cla_loss = 0.0
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch + 1, args["num_comm"]), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels, sign) in enumerate(train_loader):
        start_time = time.time()
        torch.set_grad_enabled(True)
        imagesA = imagesA.to(dev)
        labels = labels.to(dev)
        out_A = model(imagesA)
        out_A = list(out_A.values())[0][:, 1]

        loss_x = criterion(out_A, np.squeeze(labels))
        lossValue = loss_x
        lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
        # scheduler.step()
        # lr_sch.step()

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time

        cla_loss += loss_x.item()
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f} \n '
        bar.suffix = bar_str.format(step + 1, iters_per_epoch, batch_time=batch_time * (iters_per_epoch - step) / 60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch

    bar.finish()
    return lr, epoch_loss, model.state_dict()


def train_step_cls_prox(train_loader, model, epoch, optimizer, criterion, args, sever_model):

    # The hyper parameter for fedprox
    prox_mu = 0.01
    # switch to train mode
    model.train()
    epoch_loss = 0.0
    cla_loss = 0.0
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch + 1, args["num_comm"]), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, mask, labels, sign) in enumerate(train_loader):

        start_time = time.time()

        imagesA = imagesA.to(dev)
        labels = labels.to(dev)
        out_A = model(imagesA)
        out_A = list(out_A.values())[0][:, 1]
        loss_x = criterion(out_A, np.squeeze(labels))
        lossValue = loss_x
        lr = optimizer.param_groups[0]['lr']

        if step > 0:
            w_diff = torch.tensor(0., device=dev)
            for w, w_t in zip(sever_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            w_diff = torch.sqrt(w_diff)
            lossValue += prox_mu / 2. * w_diff

        optimizer.zero_grad()
        lossValue.requires_grad_(True)
        lossValue.backward()
        optimizer.step()

        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time

        cla_loss += loss_x.item()
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f} \n '
        bar.suffix = bar_str.format(step + 1, iters_per_epoch, batch_time=batch_time * (iters_per_epoch - step) / 60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch

    bar.finish()
    return lr, epoch_loss
