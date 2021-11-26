# - Predicts over texts in ../data/BalancedTweets.csv according to model
# - saved in ../models/model01
cd ../src/main/python/.
python predict.py -d ../../../data/BalancedTweets.csv -s ',' -t Tweet -l Predictions -m ../../../models/model01 -o ../../../data/Predictions.csv