import gensim
#import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from word_embedding import do_word_embedding
from batch_data import batch_data
from sklearn.model_selection import train_test_split

from nltk import word_tokenize

full_file_path = 'C:\\Documents\\Python\\Final_Project\\DataSort\\'
verbose = False

######################
# Word Embedding
######################
print('Starting word embeddings...')
#this embedding drops any words that occur less than 5 times
#word_embedding_model = do_word_embedding(full_file_path)
#word_embedding_model.save('word_embedding_model.mod')
     
#Word Embedding: load embedding from file 
wemb = gensim.models.Word2Vec.load('word_embedding_model.mod')
wemb_size = wemb['bob'].shape[0]

#Get list of keys
keys = wemb.wv.vocab
vocab_size = len(keys)

#Splits the vocab from the vector with an inbetween index
embedding_matrix= np.zeros((vocab_size,200), dtype=float)
keys_index = dict()
index = 0
for k in keys:
    #print(k)
    keys_index[k]          = index
    embedding_matrix[index] = wemb[k]
    index = index + 1
print('Done!')
######################
# Read Data - I think there's an issue with this function
######################
print('Reading Data...')
df, num_threads, largest_size = batch_data(full_file_path) 
print('Done!')

######################
# Generate Features
######################
print(type(num_threads))
print(type(largest_size))

xdata = np.zeros((num_threads,largest_size),dtype=int)
words_not_in_embedding = set()
thread_index = 0
for thread in df['Word Bodies']:
    word_index = 0
    for word in thread:
        try:
            xdata[thread_index,word_index]=keys_index[word]
            word_index = word_index+1
        except:
            words_not_in_embedding.add(word)
            #print('Word not in embedding: {}'.format(word))
    thread_index = thread_index+1

print('got here')

#Setup y data
ydata = []
for y in df['Solution Count']:
    if(y > 0):
        ydata.append(1)
    else:
        ydata.append(0)
#Convert to numpy array
ydata = np.array(ydata)



print('These words occurred too infrequently to be put into the embedding')
print(words_not_in_embedding)
#Oh baby - Convert row data into numbers with word embedding
#xdata = [wemb[row] for row in df['Word Bodies']]
print('xdata shape: {}'.format(xdata.shape))
print('ydata shape: {}'.format(ydata.shape))
print('xdata type: {}'.format(xdata[0]))
print('ydata type: {}'.format(ydata[0]))
print(ydata)
print(xdata)

######################
# Split Data
######################

#Create a 80/20 train/test split of the dataset
xtrain, xtest, ytrain, ytest = train_test_split(xdata, 
                                                ydata, 
                                                random_state = 42,
                                                test_size = 0.2)

print('Data has been split!')

######################
# Generate Model
#
#The embedding layer is a look up table. Before this layer you would need to convert each word to a number
#The number represents the index of of the word in the matrix. We've already done this in the generate features section 
#
######################

#Init - The first node in a sequential neural net must contain input_dim or input_shape
model = Sequential()

model.add(Embedding(vocab_size,
                    200,
                    weights=[embedding_matrix],
                    input_length=largest_size,
                    trainable=False))

#LSTM(128) units: Positive integer, dimensionality of the output space.
#128 is the number of 'smart neurons'.  
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32))

#Drop out - Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
model.add(Dropout(0.5))

#Get the final activation binary number          
model.add(Dense(1, activation='sigmoid'))

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary)

######################
# Fit Model
######################

#Fit the data, , iterating on the data in batches of 32 samples
model.fit(xtrain, ytrain, epochs=10, batch_size=32)

######################
# Test Model
######################
scores = model.evaluate(xtest, ytest, verbose=0)
print('Accuracy: {}'.format(scores))


