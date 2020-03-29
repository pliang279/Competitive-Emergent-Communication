# Emergent Communication in Competitive Multi-Agent Teams

> Pytorch implementation for emergent communication in competitive settings.

Correspondence to: 
  - Paul Liang (pliang@cs.cmu.edu)
  
## Paper

[**On Emergent Communication in Competitive Multi-Agent Teams**](https://arxiv.org/abs/2003.01848)<br>
[Paul Pu Liang*](http://www.cs.cmu.edu/~pliang/), Jeffrey Chen, [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/), [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/), [Satwik Kottur](https://satwikkottur.github.io/)<br>
AAMAS 2020

## Installation

All our code is implemented in PyTorch. Current version has been tested in Python 3.6 and PyTorch 1.4.

Our code also uses some python packages that can be installed as follows:

```
pip install json
pip install tqdm
pip install pickle
pip install json
pip install matplotlib
```

The next step is to clone the repository:
```bash
https://github.com/pliang279/Competitive-Emergent-Communication.git
```

## Contents

* `options.py` - Read the options from the commandline
* `dataloader.py` - Create and handle data for toy instances
* `chatbots.py` - Conversational agents - Abot and Qbot
* `learnChart.py` - Obtain evolution of language chart from checkpoints
* `html.py` - Easy creation of html tables
* `utilities.py` -  Helper functions
* `train.py` - Script to train conversational agents
* `test.py` - Script to test agents

## Usage

Please look at the scripts under `scipts/` to see how train our model.

For example, `scripts/train_111.sh` represents training a team with 3 sources of competitive influence: reward sharing, dialog overhearing, and task sharing.

`plot.py` parses all these results and plots the test accuracies across training epochs, with standard deviations highlighted

The folder `cooperative_baselines/` is largely modified from the initial cooperative settings of Task & Talk, as presented in https://github.com/batra-mlp-lab/lang-emerge.

## References

If you find this code useful, please cite our paper:

```bash
@inproceedings{liang2020_competitive,
  title={On Emergent Communication in Competitive Multi-Agent Teams},
  author={Paul Pu Liang, Jeffrey Chen, Ruslan Salakhutdinov, Louis-Philippe Morency, and Satwik Kottur},
  booktitle = {Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems, {AAMAS} '20},
  year={2020},
}
```

# Acknowledgements

This codebase was adapted from https://github.com/batra-mlp-lab/lang-emerge.
