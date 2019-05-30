import os
import time
import numpy as np
import torch
import tqdm
from utils.meters import AverageMeter, accuracy


def runTrain(model, args, trainLoader, epoch,optimizer,criterion,logging):

    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainLoader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(float(prec1), input.size(0))
        top5.update(float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            logging.info('Epoch Train: [{0}]\t'
                         'Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainLoader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def runTest(model, args, testLoader, epoch,criterion,logging):

    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(testLoader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            out = model(inputs)
            loss = criterion(out, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(float(prec1), input.size(0))
        top5.update(float(prec5), input.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)

    logging.info('Epoch Test: [{0}]\t'
          'Time ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch,  batch_time=batch_time, loss=losses,
        top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

