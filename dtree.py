# script to train interactive bots in toy world
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, pdb, random, pickle, os, json
import numpy as np
from chatbots import Team
from dataloader import Dataloader
import options
from learnChart import *
from time import gmtime, strftime

# read the command line options
options = options.read();
#------------------------------------------------------------------------
# setup experiment and dataset
#------------------------------------------------------------------------

# random seed
torch.manual_seed(options['seed']);
random.seed(options['seed']);

data = Dataloader(options);
numInst = data.getInstCount();

params = data.params;
# append options from options to params
for key, value in options.iteritems(): params[key] = value;

#------------------------------------------------------------------------
# build agents, and setup optmizer
#------------------------------------------------------------------------
team = Team(params);
team.train();
optimizer = optim.Adam([{'params': team.aBot1.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.qBot1.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.aBot2.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.qBot2.parameters(), \
                                'lr':params['learningRate']}])
#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training
numIterPerEpoch = int(np.ceil(numInst['train']/params['batchSize']));
numIterPerEpoch = max(1, numIterPerEpoch);
count = 0;
savePath = 'models/tasks_inter_jeff_%dH_%.4flr_%r_%d_%d_%r_%r_%d_%d_%.2f.pickle' %\
            (params['hiddenSize'], params['learningRate'], params['remember'],\
            options['aOutVocab'], options['qOutVocab'], options['overhear'],\
            options['overhearTask'], params['rlPosMult'], params['rlNegMult'],\
            options['overhearFraction']);

matches1 = {};
accuracy1 = {};
matches2 = {};
accuracy2 = {};
trainAccHistory1 = [];
testAccHistory1 = [];
trainAccHistory2 = [];
testAccHistory2 = [];
forest1 = [];
forest2 = [];
for iterId in xrange(params['numEpochs'] * numIterPerEpoch):
    epoch = float(iterId)/numIterPerEpoch;

    # get double attribute tasks
    if 'train' not in matches1:
        batchImg1, batchTask1, batchLabels1 \
                            = data.getBatch(params['batchSize']);
    else:
        batchImg1, batchTask1, batchLabels1 \
                = data.getBatchSpecial(params['batchSize'], matches1['train'],\
                                                        params['negFraction']);
    if 'train' not in matches2:
        batchImg2, batchTask2, batchLabels2 \
                            = data.getBatch(params['batchSize']);
    else:
        batchImg2, batchTask2, batchLabels2 \
                = data.getBatchSpecial(params['batchSize'], matches2['train'],\
                                                        params['negFraction']);

    # forward pass
    # overhear according to param overhearFraction
    overhear = options['overhear'] and random.random() < options['overhearFraction'];
    team.setOverhear(overhear)
    overhearTask = options['overhearTask'] and overhear;
    team.setOverhearTask(overhearTask)
    team.forward(Variable(batchImg1), Variable(batchTask1), Variable(batchImg2),\
        Variable(batchTask2));
    # backward pass
    team.backward(optimizer, batchLabels1, batchLabels2, epoch);

    # take a step by optimizer
    optimizer.step()
    #--------------------------------------------------------------------------
    # switch to evaluate
    team.evaluate();
    team.setOverhear(False);
    team.setOverhearTask(False);

    for dtype in ['train', 'validation']:
        # get the entire batch
        img, task, labels = data.getCompleteData(dtype);
        # evaluate on the train dataset, using greedy policy
        guess1,guess2,_,_,talk1,talk2 = team.forward(Variable(img), Variable(task),\
            Variable(img), Variable(task), record=True);
        # compute accuracy for color, shape, and both
        firstMatch1 = guess1[0].data == labels[:, 0].long();
        secondMatch1 = guess1[1].data == labels[:, 1].long();
        matches1[dtype] = firstMatch1 & secondMatch1;
        accuracy1[dtype] = 100*torch.sum(matches1[dtype]).float()\
                                    /float(matches1[dtype].size(0));
        firstMatch2 = guess2[0].data == labels[:, 0].long();
        secondMatch2 = guess2[1].data == labels[:, 1].long();
        matches2[dtype] = firstMatch2 & secondMatch2;
        accuracy2[dtype] = 100*torch.sum(matches2[dtype]).float()\
                                    /float(matches2[dtype].size(0));
        # Build dialog trees
        if dtype == 'train' and iterId % 1000 == 0:
            talk1 = data.reformatTalk(talk1, guess1, img, task, labels);
            talk2 = data.reformatTalk(talk2, guess2, img, task, labels);
            savePath1 = savePath.replace('inter', 'chatlog1').replace('pickle', 'json');
            savePath2 = savePath.replace('inter', 'chatlog2').replace('pickle', 'json');
            with open(savePath1, 'w') as fileId:json.dump(talk1, fileId);
            with open(savePath2, 'w') as fileId:json.dump(talk2, fileId);
            forest1.append(buildDialogTree(savePath1));
            forest2.append(buildDialogTree(savePath2));
    # switch to train
    team.train();

    # break if train accuracy reaches 100%
    if accuracy1['train'] == 100 or accuracy2['train'] == 100: break;

    # save for every 5k epochs
    if iterId > 0 and iterId % (5000*numIterPerEpoch) == 0:
        team.saveModel(savePath, optimizer, params);
        historySavePath = savePath.replace('inter', 'history');
        with open(historySavePath, 'wb') as f:
            pickle.dump({
                    'train1': trainAccHistory1,
                    'valid1': testAccHistory1,
                    'train2': trainAccHistory2,
                    'valid2': testAccHistory2
                }, f);

    if iterId % 100 != 0: continue;

    time = strftime("%a, %d %b %Y %X", gmtime());
    print('[%s][Iter: %d][Ep: %.2f][R1: %.4f][Tr1: %.2f Va1: %.2f]' % \
                                (time, iterId, epoch, team.totalReward1,\
                                accuracy1['train'], accuracy1['validation']))
    print('[%s][Iter: %d][Ep: %.2f][R2: %.4f][Tr2: %.2f Va2: %.2f]' % \
                                (time, iterId, epoch, team.totalReward2,\
                                accuracy2['train'], accuracy2['validation']))
    trainAccHistory1.append(accuracy1['train'].data.item());
    testAccHistory1.append(accuracy1['validation'].data.item());
    trainAccHistory2.append(accuracy2['train'].data.item());
    testAccHistory2.append(accuracy2['validation'].data.item());

#------------------------------------------------------------------------
print('[%s][Iter: %d][Ep: %.2f][R1: %.4f][Tr1: %.2f Va1: %.2f]' % \
			(time, iterId, epoch, team.totalReward1,\
			accuracy1['train'], accuracy1['validation']))
print('[%s][Iter: %d][Ep: %.2f][R2: %.4f][Tr2: %.2f Va2: %.2f]' % \
			(time, iterId, epoch, team.totalReward2,\
			accuracy2['train'], accuracy2['validation']))
# save final model with a time stamp
timeStamp = strftime("%a-%d-%b-%Y-%X", gmtime());
replaceWith = 'final_%s' % timeStamp;
finalSavePath = savePath.replace('inter', replaceWith);
print('Saving : ' + finalSavePath)
team.saveModel(finalSavePath, optimizer, params);
#------------------------------------------------------------------------
historySavePath = finalSavePath.replace('final', 'history')
with open(historySavePath, 'wb') as f:
    pickle.dump({
            'train1': trainAccHistory1,
            'valid1': testAccHistory1,
            'train2': trainAccHistory2,
            'valid2': testAccHistory2
        }, f)

backtrackLanguageChart(forest1);
backtrackLanguageChart(forest2);
