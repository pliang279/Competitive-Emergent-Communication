DATA_PATH='data/toy64_split_0.8-0.1.json'
VISIBLE_CUDA_DEVICES=0
# script to run the program
for posMult in 2 10 50
do for negMult in 2 10 50
do python train.py -learningRate 0.01 -hiddenSize 100 -batchSize 1000 \
                    -imgFeatSize 20 -embedSize 20\
                    -useGPU -dataset $DATA_PATH\
                    -aOutVocab 4 -qOutVocab 3 -overhear -numEpochs 50001\
                    -rlPosMult $posMult -rlNegMult $negMult\
                    -overhearFraction 0.5
done
done
