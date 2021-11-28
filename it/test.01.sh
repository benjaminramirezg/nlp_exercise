# - Reads the unbalanced ../data/Tweets.csv dataset and
#   creates a balanced version by undersampling the less populated class
# - Splits the dataset into training, validation and testing sets
# - Saves the four resulting datasets in ../data/dataset/

cd ../src/main/python/.
python prepare_dataset.py -d ../../../data/Tweets.csv -s ',' -t 'Tweet' -l 'ISIS Flag' -p 0.80 -o ../../../data/dataset/
