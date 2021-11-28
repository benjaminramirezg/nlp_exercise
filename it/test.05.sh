# - Predicts over texts in ../data/dataset/testing_set.csv according to model
# - saved in ../models/model_mlp01
cd ../src/main/python/.
python predict.py -d ../../../data/dataset/testing_set.csv -s ',' -t Tweet -l Predictions -m ../../../models/model_lr01 -o ../../../data/dataset/predictions_mlp01.csv