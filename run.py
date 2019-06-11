import time

import numpy as np
import torch

from utils.meters import AverageMeter, accuracy


def runTrain(model, args, trainLoader, epoch, optimizer, criterion, logging, use_corr=False):
    model.train()
    batch_time = AverageMeter()
    totalLosses = AverageMeter()
    ceLosses = AverageMeter()
    corrLosses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        out = model(inputs)
        # for m in model.modules():
        #     if hasattr(m,"corr"):
        #         if 'corr' in locals():
        #             corr = torch.cat((corr, m.corr))
        #         else:
        #             corr = m.corr
        corr = torch.sum(torch.stack([m.corr for m in model.modules() if hasattr(m, "corr")]))
        totalLoss, crossEntropyLoss, corrLoss = criterion(out, targets, corr)
        if use_corr:
            ls = totalLoss
        else:
            ls = crossEntropyLoss
        ls.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        totalLosses.update(totalLoss.item(), inputs.size(0))
        ceLosses.update(crossEntropyLoss.item(), inputs.size(0))
        corrLosses.update(corrLoss.item(), inputs.size(0))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            logging.info('Epoch Train: [{}]\t'
                         'Train: [{}/{}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Cross Entropy Loss {CEloss.val:.4f} ({CEloss.avg:.4f})\t'
                         'Correlation Loss {Corrloss.val:.4f} ({Corrloss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx + 1, len(trainLoader), batch_time=batch_time, loss=totalLosses,
                CEloss=ceLosses, Corrloss=corrLosses, top1=top1, top5=top5))

    return totalLosses.avg, ceLosses.avg, corrLosses.avg, top1.avg, top5.avg


def runTest(model, args, testLoader, epoch, criterion, logging):
    model.eval()
    batch_time = AverageMeter()
    totalLosses = AverageMeter()
    ceLosses = AverageMeter()
    corrLosses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    entropy = 0
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testLoader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            out = model(inputs)
            corr = torch.sum(torch.tensor([m.corr for m in model.modules() if hasattr(m, "corr")]))
            totalLoss, crossEntropyLoss, corrLoss = criterion(out, targets, corr)

            entropy += np.sum(np.array([x.bit_count for x in model.modules() if hasattr(x, "bit_count")]))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        totalLosses.update(totalLoss.item(), inputs.size(0))
        ceLosses.update(crossEntropyLoss.item(), inputs.size(0))
        corrLosses.update(corrLoss.item(), inputs.size(0))
        top1.update(float(prec1), inputs.size(0))
        top5.update(float(prec5), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    act_count = np.sum(np.array([x.act_size for x in model.modules() if hasattr(x, "act_size")]))
    avgEntropy = float(entropy) / len(testLoader) / act_count
    logging.info('Epoch Test: [{}]\t'
                 'Time ({batch_time.avg:.3f})\t'
                 'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'Cross Entropy Loss {CEloss.val:.4f} ({CEloss.avg:.4f})\t'
                 'Correlation Loss {Corrloss.val:.4f} ({Corrloss.avg:.4f})\t'
                 'Entropy {ent} \t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, batch_time=batch_time, loss=totalLosses,
        CEloss=ceLosses, Corrloss=corrLosses, ent=avgEntropy, top1=top1, top5=top5))

    return totalLosses.avg, ceLosses.avg, corrLosses.avg, top1.avg, top5.avg, avgEntropy
