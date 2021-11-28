# - Trains model lr01 given the dataset in ../data/dataset/training_set.csv
#   and the config in ../config/config.lr01.json. Saves the resulting artifacts
# - in ../models/modellr01
cd ../src/main/python/.
python train.py -d ../../../data/dataset/training_set.csv -v ../../../data/dataset/validation_set.csv -s ',' -t 'Tweet' -l 'ISIS Flag' -o ../../../models/modellr01 -c ../../../config/config.lr01.json
