from __future__ import print_function

import argparse
import logging
import os
import random
import time
from inspect import getfile, currentframe
from os import getpid, environ
from os.path import dirname, abspath
from socket import gethostname
from sys import exit, argv

import mlflow
import torch
import torch.backends.cudnn as cudnn
from torch import manual_seed as torch_manual_seed
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange

import Models
from corrLoss import corrLoss
from run import runTrain, runTest
from utils.dataset import loadModelNames, saveArgsToJSON, TqdmLoggingHandler, load_data, loadDatasets


def parseArgs():
    modelNames = loadModelNames()
    datasets = loadDatasets()

    parser = argparse.ArgumentParser(description='Transform Coding')
    # general
    parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=datasets.keys(), help='dataset name')

    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch', default=250, type=int, help='batch size')
    parser.add_argument('--exp', type=str, default='', help='experiment name')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 2)')
    parser.add_argument('--print_freq', default=50, type=int, help='Number of batches between log messages')
    parser.add_argument('--pre-trained', default='preTrained', type=str, help='location of the pretrained models')
    parser.add_argument('--onlyInference', action='store_true', help='If use only inference')
    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='The learning rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 50, 100],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--regul', type=float, default=0, help='Regularization strength')

    # algorithm
    parser.add_argument('--actBitwidth', default=32, type=float,
                        help='Quantization activation bitwidth (default: 32)')
    parser.add_argument('--weightBitwidth', default=32, type=int,
                        help='Quantization weight bitwidth (default: 32)')
    parser.add_argument('--model', '-a', metavar='MODEL', required=True, choices=modelNames,
                        help='model architecture: ' + ' | '.join(modelNames))
    parser.add_argument('--MicroBlockSz', type=int, default=1, help='Size of block in H*W')
    parser.add_argument('--channelsDiv', type=int, default=1, help='How many parts divide the number of channels')
    parser.add_argument('--eigenVar', type=float, default=1.0, help='EigenVar - should be between 0 to 1')
    parser.add_argument('--transformType', type=str, default='eye', choices=['eye', 'pca', 'pcaT', 'pcaQ'],
                        help='which projection we do: [eye, pca, pcaQ, pcaT]')
    parser.add_argument('--transform', action='store_true',
                        help='if use linear transformation, otherwise use regular inference')
    parser.add_argument('--use-corr', action='store_true', help='Use correlation loss')
    parser.add_argument('--ea', action='store_true', help='Use entropy approximation')
    parser.add_argument('--ei', action='store_true', help='Minimize entropy')
    parser.add_argument('--ea-lr', type=float, default=1e-3, help='entropy approximation lr')
    parser.add_argument('--ei-lr', type=float, default=1e-3, help='entropy minimization lr')

    args = parser.parse_args()
    args.nClasses = datasets[args.dataset]

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        assert (args.model == 'resnet20' or args.model == 'resnet56')

    return args


if __name__ == '__main__':

    args = parseArgs()

    # set number of model output classes

    # run only in GPUs
    if not is_available():
        print('no gpu device available')
        exit(1)

    # update GPUs list
    if type(args.gpu) is not 'None':
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    # CUDA
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    set_device(args.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)

    # create folder
    baseFolder = dirname(abspath(getfile(currentframe())))
    args.time = time.strftime("%Y%m%d-%H%M%S")
    args.folderName = '{}_{}_{}_{}_{}_{}_{}'.format(args.model, args.transformType, args.actBitwidth,
                                                    args.weightBitwidth, args.MicroBlockSz,
                                                    args.channelsDiv, args.time)
    args.save = '{}/results/{}'.format(baseFolder, args.folderName)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # save args to JSON
    saveArgsToJSON(args)

    # Logger
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename=os.path.join(args.save, 'log.txt'), level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    th = TqdmLoggingHandler()
    th.setFormatter(logging.Formatter(log_format))
    log = logging.getLogger()
    log.addHandler(th)

    # Data
    testLoader, trainLoader = load_data(args, logging)

    # Model
    logging.info('==> Building model..')
    modelClass = Models.__dict__[args.model]
    model = modelClass(args)

    # Load preTrained weights.
    logging.info('==> Resuming from checkpoint..')
    model.loadPreTrained(args.pre_trained, 'cpu')
    model = model.cuda()

    criterion = corrLoss(args).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    # log command line
    logging.info('CommandLine: {} PID: {} '
                 'Hostname: {} CUDA_VISIBLE_DEVICES {}'.format(argv, getpid(), gethostname(),
                                                               environ.get('CUDA_VISIBLE_DEVICES')))

    # # Weights quantization
    # if args.weightBitwidth < 32 and not args.fold:
    #     model_path = './qmodels'
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)
    #     model_path = os.path.join(model_path, args.model + ('_kmeans%dbit.pt' % args.weightBitwidth))
    #     if not os.path.exists(model_path):
    #         model = quantizeWeights(model, args.weightBitwidth, logging)
    #         torch.save(model, model_path)
    #     else:
    #         torch.load(model_path)
    #         logging.info('Loaded preTrained model with weights quantized to {} bits'.format(args.weightBitwidth))

    # mlflow
    mlflow.set_tracking_uri(os.path.join(baseFolder, 'mlruns_mxt'))

    mlflow.set_experiment(args.exp + '_' + args.model)
    with mlflow.start_run(run_name="{}".format(args.folderName)):
        params = vars(args)
        for p in params:
            mlflow.log_param(p, params[p])
        start_epoch = 0
        minCorrLoss = 10000

        for epoch in trange(start_epoch, args.epochs):
            scheduler.step()
            out = runTest(model, args, testLoader, epoch, criterion, logging)
            testTotalLoss, testCELoss, testCorrLoss, testTop1, testTop5, avgEntropy = out
            if not args.onlyInference:
                out = runTrain(model, args, trainLoader, epoch, optimizer, criterion, logging, args.use_corr)
                trainTotalLoss, trainCELoss, trainCorrLoss, trainTop1, trainTop5 = out

            if mlflow.active_run() is not None:
                mlflow.log_metric('Test top1', testTop1)
                mlflow.log_metric('Test top5', testTop5)
                if not args.onlyInference:
                    mlflow.log_metric('Train top1', trainTop1)
                    mlflow.log_metric('Train top5', trainTop5)
                    mlflow.log_metric('trainTotalLoss', trainTotalLoss)
                    mlflow.log_metric('trainCELoss', trainCELoss)
                    mlflow.log_metric('trainCorrLoss', trainCorrLoss)
                mlflow.log_metric('TestTotalloss', testTotalLoss)
                mlflow.log_metric('testCELoss', testCELoss)
                mlflow.log_metric('testCorrLoss', testCorrLoss)
                mlflow.log_metric('avgEntropy', avgEntropy)
            # Save checkpoint.
            if minCorrLoss > testCorrLoss:
                state = {
                    'net': model.state_dict(),
                    'acc': testTop1,
                    'epoch': epoch,
                    'testCorrLoss': testCorrLoss,
                }
                torch.save(state, args.save + '/ckpt.t7')
                minCorrLoss = testCorrLoss
