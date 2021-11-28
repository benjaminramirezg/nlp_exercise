# - Trains model mlp01 given the dataset in ../data/dataset/training_set.csv
#   and the config in ../config/config.mlp01.json. Saves the resulting artifacts
# - in ../models/modelmlp01
cd ../src/main/python/.
python train.py -d ../../../data/dataset/training_set.csv -v ../../../data/dataset/validation_set.csv -s ',' -t 'Tweet' -l 'ISIS Flag' -o ../../../models/modelmlp01 -c ../../../config/config.mlp01.json
