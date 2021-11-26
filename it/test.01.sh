# - Reads the unbalanced ../data/Tweets.csv dataset and
# - creates a balanced version by undersampling the less populated class
cd ../scripts/.
python undersample.py -d ../data/Tweets.csv -s ',' -o ../data/BalancedTweets.csv -t Tweet -l 'ISIS Flag'
