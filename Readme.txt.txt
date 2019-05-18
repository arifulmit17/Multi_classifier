First open the file Multi_classifier.py

import multi_classifier_rnn.py file
import preprocess.py file

give file path to pd.read_csv('training data file path')
for example,
train_data=pd.read_csv('training_data_double.csv')

save and close the file.

Open the file multi_classifier_rnn.py 

give file path to pd.read_csv('test data file path')

for example,
test_data=pd.read_csv('test_1.csv')

to use pretrained word2vec embeddings put file path in line,
for example,

word_vectors = KeyedVectors.load_word2vec_format('C:\\Users\\User\\Documents\\resources\\Multi_classifier\\GoogleNews-vectors-negative300.bin', binary=True)

save and close the file

the GoogleNews-vectors-negative300.bin can be downloaded from,

http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/


