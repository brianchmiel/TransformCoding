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
    eLosses = AverageMeter()
    eiLosses = AverageMeter()
    leLosses = AverageMeter()
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
        if args.ea:
            eloss = None
            eiloss = None
            leloss = None
            cnt = 0
            for layer in model.modules():
                if hasattr(layer, 'entropy_loss_value'):
                    cnt += 1
                    if eloss is None:
                        eloss = layer.entropy_loss_value
                    else:
                        eloss += layer.entropy_loss_value
                    if batch_idx % args.print_freq == 0:
                        print(cnt, layer.entropy_loss_value.item())
                    leloss = layer.entropy_loss_value.item()
                    if args.ei:
                        if eiloss is None:
                            eiloss = layer.entropy_value.mean()
                        else:
                            eiloss += layer.entropy_value.mean()

            if eloss is not None:
                ls += 1e-3 * eloss  # TODO: parameter
                eLosses.update(eloss.item(), inputs.size(0))
                leLosses.update(leloss, inputs.size(0))
            if eiloss is not None:
                ls += 1e-5 * eiloss  # TODO: parameter
                eiLosses.update(eiloss.item(), inputs.size(0))
        ls.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out, targets, topk=(1, 5))
        totalLosses.update(ls.item(), inputs.size(0))
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
                         'Entropy Loss {Eloss.val:.4f} ({Eloss.avg:.4f})\t'
                         'Last layer entropy MSE {lEloss.val:.4f} ({lEloss.avg:.4f})\t'
                         'Entropy I Loss {EIloss.val:.4f} ({EIloss.avg:.4f})\t'
                         'Correlation Loss {Corrloss.val:.4f} ({Corrloss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx + 1, len(trainLoader), batch_time=batch_time, loss=totalLosses, CEloss=ceLosses,
                lEloss=leLosses, EIloss=eiLosses, Eloss=eLosses, Corrloss=corrLosses, top1=top1, top5=top5))

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
