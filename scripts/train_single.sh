# CUDA_VISIBLE_DEVICES=3 nohup python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 500000 -seed 0 > res/rs0_do0_ts0.txt &
# CUDA_VISIBLE_DEVICES=3 nohup python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 500000 -seed 0 -rshare > res/rs1_do0_ts0.txt &
# CUDA_VISIBLE_DEVICES=2 nohup python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 500000 -seed 0 -overhear > res/rs0_do1_ts0.txt &
# CUDA_VISIBLE_DEVICES=2 nohup python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 500000 -seed 0 -overhearTask > res/rs0_do0_ts1.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 500000 -seed 0 -rshare -overhear > res/rs1_do1_ts0.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 500000 -seed 0 -rshare -overhearTask > res/rs1_do0_ts1.txt &
# CUDA_VISIBLE_DEVICES=0 nohup python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 500000 -seed 5 -overhear -overhearTask > res/rs0_do1_ts1_v2.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train.py -useGPU -dataset 'data/toy64_split_0.8-0.1.json' -numEpochs 500000 -seed 123 -rshare -overhear -overhearTask > res/rs1_do1_ts1_v6.txt &