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
from eval import all_metrics

#------------------------------------------------------------------------
# load experiment and model
#------------------------------------------------------------------------
#if len(sys.argv) < 2:
#    print('Wrong usage:')
#    print('python test.py <modelPath>')
#    sys.exit(0);


def do_test(loadPath):
    print('Loading model from: %s' % loadPath)
    with open(loadPath, 'r') as fileId:
        loaded = pickle.load(fileId);

    #------------------------------------------------------------------------
    # build dataset, load agents
    #------------------------------------------------------------------------
    params = loaded['params'];
    data = Dataloader(params);

    team = Team(params);
    team.loadModel(loaded);
    team.evaluate();
    #------------------------------------------------------------------------
    # test agents
    #------------------------------------------------------------------------
    dtypes = ['train']
    for dtype in dtypes:
        # evaluate on the train dataset, using greedy policy
        images, tasks, labels = data.getCompleteData(dtype);
        # forward pass
        preds, _, talk, talk_list = team.forward(Variable(images), Variable(tasks), True);

        options = dict()
        options['qOutVocab'] = 3
        options['aOutVocab'] = 4
        m1,m2,ic1,ic2,h1,h2 = all_metrics(team, preds, talk_list, options)

        # compute accuracy for first, second and both attributes
        firstMatch = preds[0].data == labels[:, 0].long();
        secondMatch = preds[1].data == labels[:, 1].long();
        matches = firstMatch & secondMatch;
        atleastOne = firstMatch | secondMatch;

        # compute accuracy
        firstAcc = 100 * torch.mean(firstMatch.float());
        secondAcc = 100 * torch.mean(secondMatch.float());
        atleastAcc = 100 * torch.mean(atleastOne.float());
        accuracy = 100 * torch.mean(matches.float());
        print('\nOverall accuracy [%s]: %.2f (f: %.2f s: %.2f, atleast: %.2f)'\
                        % (dtype, accuracy, firstAcc, secondAcc, atleastAcc));

        # pretty print
        talk = data.reformatTalk(talk, preds, images, tasks, labels);
        if 'final' in loadPath:
            savePath = loadPath.replace('final', 'chatlog-'+dtype);
        elif 'inter' in loadPath:
            savePath = loadPath.replace('inter', 'chatlog-'+dtype);
        savePath = savePath.replace('pickle', 'json');
        print('Saving conversations: %s' % savePath)
        with open(savePath, 'w') as fileId: json.dump(talk, fileId);
        saveResultPage(savePath);

        res1 = accuracy, firstAcc, secondAcc, atleastAcc
        res2 = m1,m2,ic1,ic2,h1,h2
        return res1, res2


# load and compute on test
#loadPath = sys.argv[1];
#loadDir = '/media/bighdd4/Paul/emergent_language/old/lang-emerge/models/'
#loadDir = '/media/bighdd4/Paul/emergent_language/old/lang-emerge/models2'
loadDir = '/media/bighdd4/Paul/emergent_language/lang-emerge/models/'
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(loadDir) if isfile(join(loadDir, f))]
accs,fs,ss,ats,m1s,m2s,ic1s,ic2s,h1s,h2s = [],[],[],[],[],[],[],[],[],[]
mall,icall,hall = [],[],[]
for file in onlyfiles:
    if 'tasks_final' in file:
        loadPath = join(loadDir, file)
        res1, res2 = do_test(loadPath)
        acc,f,s,at = res1
        m1,m2,ic1,ic2,h1,h2 = res2
        accs.append(acc.item())
        fs.append(f.item())
        ss.append(s.item())
        ats.append(at.item())
        
        m1s.append(m1.item())
        m2s.append(m2.item())

        mall.append(m1.item())
        mall.append(m2.item())

        ic1s.append(ic1.item())
        ic2s.append(ic2.item())

        icall.append(ic1.item())
        icall.append(ic2.item())

        h1s.append(h1)
        h2s.append(h2)

        hall.append(h1)
        hall.append(h2)

print ()
print ('accs', sum(accs)/float(len(accs)), np.std(np.array(accs)))
print ('fs', sum(fs)/float(len(fs)), np.std(np.array(fs)))
print ('ss', sum(ss)/float(len(ss)), np.std(np.array(ss)))
print ('ats', sum(ats)/float(len(ats)), np.std(np.array(ats)))
print ('m1s', sum(m1s)/float(len(m1s)), np.std(np.array(m1s)))
print ('m2s', sum(m2s)/float(len(m2s)), np.std(np.array(m2s)))
print ('ic1s', sum(ic1s)/float(len(ic1s)), np.std(np.array(ic1s)))
print ('ic2s', sum(ic2s)/float(len(ic2s)), np.std(np.array(ic2s)))
print ('h1s', sum(h1s)/float(len(h1s)), np.std(np.array(h1s)))
print ('h2s', sum(h2s)/float(len(h2s)), np.std(np.array(h2s)))
print ()
print ('mall', sum(mall)/float(len(mall)), np.std(np.array(mall)))
print ('icall', sum(icall)/float(len(icall)), np.std(np.array(icall)))
print ('hall', sum(hall)/float(len(hall)), np.std(np.array(hall)))






