# script to develop a toy example
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, pdb, random, pickle, json
import numpy as np
from chatbots import Team
from dataloader import Dataloader

import sys
sys.path.append('../');
from utilities import saveResultPage

#------------------------------------------------------------------------
# load experiment and model
#------------------------------------------------------------------------
if len(sys.argv) < 2:
    print('Wrong usage:')
    print('python test.py <modelPath>')
    sys.exit(0);

# load and compute on test
loadPath = sys.argv[1];
print('Loading model from: %s' % loadPath)
with open(loadPath, 'r') as fileId: loaded = pickle.load(fileId);

#------------------------------------------------------------------------
# build dataset, load agents
#------------------------------------------------------------------------
params = loaded['params'];
data = Dataloader(params);

team = Team(params);
team.loadModel(loaded);
team.evaluate();
team.setOverhear(False);
team.setOverhearTask(False);
#------------------------------------------------------------------------
# test agents
#------------------------------------------------------------------------
# evaluate on the train dataset, using greedy policy
images = itertools.product((0,1,2,3),(4,5,6,7),(8,9,10,11))
tasks = (0,1,2,3,4,5)
alldata = list(itertools.product(images, tasks))
images = torch.LongTensor([x[0] for x in alldata])
tasks = torch.LongTensor([x[1] for x in alldata])
selectInds = data.taskSelect[tasks]
if data.useGPU:
    images = images.cuda()
    tasks = tasks.cuda()
    selectInds = selectInds.cuda()
labels = images.gather(1, selectInds)

# forward pass
preds1,preds2,_,_,talk1,talk2 = team.forward(Variable(images),\
Variable(tasks), Variable(images), Variable(tasks), True);

# pretty print
talk1 = data.reformatTalk(talk1, preds1, images, tasks, labels);
talk2 = data.reformatTalk(talk2, preds2, images, tasks, labels);
if 'final' in loadPath:
    savePath = loadPath.replace('final', 'chatlog');
elif 'inter' in loadPath:
    savePath = loadPath.replace('inter', 'chatlog');
savePath1 = savePath.replace('.pickle', '_1.json');
savePath2 = savePath.replace('.pickle', '_2.json');
print('Saving conversations: %s' % savePath)
with open(savePath1, 'w') as fileId:json.dump(talk1, fileId);
with open(savePath2, 'w') as fileId:json.dump(talk2, fileId);
saveResultPage(savePath1);
saveResultPage(savePath2);
