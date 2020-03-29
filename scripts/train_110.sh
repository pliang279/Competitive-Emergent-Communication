DATA_PATH='data/toy64_split_0.8.json'
# script to run the program
for i in $(seq 0 9); do CUDA_VISIBLE_DEVICES=1 python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 100000 -seed 0 -rlPosMult 1 -rlNegMult 10 -rshare -overhear; done