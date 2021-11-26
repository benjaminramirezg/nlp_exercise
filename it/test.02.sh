# - Trains model v.01 given the balanced dataset in ../data/BalancedTweets.csv
# - and the config in ../config/config.01.json. Saves the resulting artifacts
# - in ../models/model01
cd ../src/main/python/.
python train.py -d ../../../data/BalancedTweets.csv -s ',' -t 'Tweet' -l 'ISIS Flag' -o ../../../models/model01 -c ../../../config/config.01.json
