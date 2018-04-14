#libraries
import re
import json
import os
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import io
from filename import get_my_file_path
import ast
from nltk.tokenize.moses import MosesTokenizer
from nltk.corpus import stopwords
from nltk import bigrams
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse import hstack

    
    
#define global variable for paths and data files
folder_path = get_my_file_path()
mosesTokenizer = MosesTokenizer()
stop_words = set(stopwords.words('english'))
more_stop_words = [',','&quot;','(',')','/','&apos;t','&apos;re','&apos;s','&apos;ve','&gt;','+','~','-','*','\\',':','--', '\'',
                   '#','$','%','&amp;','&apos;','&apos;d','&apos;ll','&apos;m','..','...','....','"']
punct_stop_words = ['?','.']
all_stop_words = stop_words.union(more_stop_words)
all_stop_words_punct = all_stop_words.union(punct_stop_words)

n_features = 1000

def splitListToRows(row,row_accumulator,target_column):
    split_row = ast.literal_eval(row[target_column])
    for s in split_row:
        s = unidecode(s)
        new_row = row.to_dict()
        new_row[target_column] = s
        row_accumulator.append(new_row)

#tokenize a message and remove stop words
def splitTokensToRows(row,row_accumulator,target_column):
    split_row = mosesTokenizer.tokenize(row[target_column])
    for s in split_row:
        s_lower = s.lower()
        if not s_lower in all_stop_words:
            new_row = row.to_dict()
            new_row[target_column] = s_lower
            row_accumulator.append(new_row)
            
#tokenize a message to bigrams and remove stop words
def splitBigramsToRows(row,row_accumulator,target_column):
    split_row = bigrams(mosesTokenizer.tokenize(row[target_column]))
    for s in split_row:
        s_lower0 = s[0].lower()
        s_lower1 = s[1].lower()
        if ((not s_lower0 in all_stop_words_punct) and (not s_lower1 in all_stop_words_punct)):
            new_row = row.to_dict()
            
            #to put them in two separate columns
            #del new_row[target_column]
            #new_row['word 0'] = s_lower[0]
            #new_row['word 0'] = s_lower[0]
            
            #to put them in the same column
            new_row[target_column] = ' '.join([s_lower0, s_lower1])
            
            row_accumulator.append(new_row)  

def main():
    #start by loading all data from the folder of files
    #declare a dataframe to build the data into
    thread_df = pd.DataFrame()
    
    all_files = os.listdir(folder_path)
    #wrap the iterator in tqdm for a progress bar
    all_files = tqdm(all_files)
    
    #iterate through all files in the folder
    for file_name in all_files:
        #Get file data, put into data frame
        full_file_location = folder_path + file_name
        with open(full_file_location, 'r') as f:
            str1 = f.read()
        doc_df = pd.read_csv(io.StringIO(str1))
        #print(doc_df.head())
        thread_df = thread_df.append(doc_df, ignore_index= True)
        
    print ("Final threads df:")
    print (list(thread_df))
    print (thread_df.head(n=5))
    #print (thread_df['Message Bodies'][1])
    
    #remove odd whitespace chars from unicode that will create odd tokens
    thread_df['Message Bodies'] = [w.replace('\\xa0', ' ').replace('\\n', ' ') for w in thread_df['Message Bodies']]
    #print (thread_df['Message Bodies'][1])
    
    #Find how many threads are marked solved
    num_solved = sum(thread_df['Solution Count']>0)
    num_threads = len(thread_df)
    percent_solved = (num_solved/num_threads)*100
    print(f"{num_solved} of {num_threads} threads are solved, or {percent_solved}%")
    
    #Create X and y data
    thread_X = thread_df.drop(columns=['Solution Count', 'Thread ID', 'Message List', 'User List', 'Message HTML', 'Manual Solve', 'Post Times', 'Message Bodies'])
    print (list(thread_X))
    print (thread_X.head(n=5))
    thread_X_csr = csr_matrix(thread_X.values.astype(int))
    print(thread_X_csr)
    thread_y = thread_df['Solution Count']
    # print (type(thread_y))
    # print (thread_y)
    
    #create a dataframe of the messages, for tokenizing
    #message_df = thread_df[['Thread ID','Message Bodies']]
    #print (list(message_df))
    #print (message_df.head(n=20))
    #print (type(message_df['Message Bodies'][0]))

    #split the list of messages into rows
    #new_rows = []
    #target_column = 'Message Bodies'
    #message_df.apply(splitListToRows,axis=1,args = (new_rows,target_column))
    #print ("new rows acquired")
    #print (type(new_rows))
    #print (new_rows[0:10])
    #new_msg_df = pd.DataFrame(new_rows)
    #print (new_msg_df['Message Bodies'][0])
    
    # Use tf-idf features
    print("Extracting tf-idf features...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       strip_accents = 'unicode',
                                       #tokenizer = mosesTokenizer.tokenize(thread_df['Message Bodies']),
                                       stop_words=all_stop_words_punct)
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(thread_df['Message Bodies'])
    print("done in %0.3fs." % (time() - t0))
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print (tfidf_feature_names)
    print (type(tfidf))
    
    tfidf_thread_X = hstack((thread_X_csr, tfidf))
    print (tfidf_thread_X)
    
    # Use tf (raw term count)
    # print("Extracting tf features...")
    # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
    #                                 max_features=n_features,
    #                                 strip_accents = 'unicode',
    #                                 #tokenizer = mosesTokenizer.tokenize(thread_df['Message Bodies']),
    #                                 ngram_range = (1,3),
    #                                 stop_words=all_stop_words_punct)
    # t0 = time()
    # tf = tf_vectorizer.fit_transform(thread_df['Message Bodies'])
    # print("done in %0.3fs." % (time() - t0))
    # tf_feature_names = tf_vectorizer.get_feature_names()
    # print (tf_feature_names)
    # 
    #Create 70-30 splits
    Xtrain, Xtest, ytrain, ytest = train_test_split(tfidf_thread_X, 
                                                    thread_y, 
                                                    random_state=42, 
                                                    train_size=.7, 
                                                    test_size=.3)
    
    print('Baseline')
    print(time())
    curModel = RandomForestClassifier().fit(Xtrain, ytrain)
    print(time())
    curTrainAccuracy = curModel.score(Xtrain, ytrain)     
    print("Train accuracy score, baseline: {:.8f}".format(curTrainAccuracy))
    print(time())
    
if __name__ == '__main__': main()