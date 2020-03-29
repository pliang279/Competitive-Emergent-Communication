DATA_PATH='data/toy64_split_0.8.json'
# script to run the program
for i in $(seq 0 9); do CUDA_VISIBLE_DEVICES=0 python train.py -learningRate 0.01 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -useGPU -dataset $DATA_PATH\
                -aOutVocab 64 -qOutVocab 64 -numEpochs 100000 -hiddenSize 50 -seed $i; done