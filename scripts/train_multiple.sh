for i in $(seq 0 9); do CUDA_VISIBLE_DEVICES=3 python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 50000 -seed $i -overhear; done
