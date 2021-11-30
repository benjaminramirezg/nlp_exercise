# - Predicts over texts in ../data/dataset/testing_set.csv according to model
# - saved in ../models/modellr01
cd ../src/main/python/.
python predict.py -d ../../../data/dataset/testing_set.csv -s ',' -t Tweet -l Predictions -m ../../../models/modellr01 -o ../../../data/dataset/predictions_lr01.csv