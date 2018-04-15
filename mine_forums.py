#libraries
import re
import os
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import io
import string
from filename import get_my_file_path
from nltk.tokenize.moses import MosesTokenizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from collections import Counter
from imblearn.under_sampling import NeighbourhoodCleaningRule 
  
#define global variable for paths and data files
folder_path = get_my_file_path()
mosesTokenizer = MosesTokenizer()
stop_words = set(stopwords.words('english'))
more_stop_words = [',','&quot;','(',')','/','&apos;t','&apos;re','&apos;s','&apos;ve','&gt;','+','~','-','*','\\',':','--', '\'',
                   '#','$','%','&amp;','&apos;','&apos;d','&apos;ll','&apos;m','..','...','....','"']
punct_stop_words = ['?','.']
all_stop_words = stop_words.union(more_stop_words)
all_stop_words_punct = all_stop_words.union(punct_stop_words)

n_features = 2000

class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
        #Part-Of-Speech Tagged and Word Tokenized 
        tagged = pos_tag((word_tokenize(doc)))
        lems = []

        #For each tagged word, lemmatize the nouns, verbs, and adjectives
        for w,t in tagged:

            ## { Part-of-speech constants
            ## ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
            ## }

            #temporay variable to potentially change the word
            l = w 
            #noun
            if(t[0] == 'N'):
                l = self.wnl.lemmatize(w, 'n')
            #verb
            elif(t[0] == 'V'):
                l = self.wnl.lemmatize(w, 'v')
            #adjective    
            elif(t[0] == 'J'):
                l = self.wnl.lemmatize(w, 'a')
    
            lems.append(l)    
            # if(l != w):
            #     print('{} {} {}'.format(w,t,l))

        #return list of lemmed words
        return lems

def train_basic_rf(Xdata, ydata):
    #Create 70-30 splits
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, 
                                                    ydata, 
                                                    random_state=42, 
                                                    train_size=.7, 
                                                    test_size=.3)
    
    # print('Baseline')
    curModel = RandomForestClassifier().fit(Xtrain, ytrain)
    startTime = time()
    curTrainAccuracy = curModel.score(Xtest, ytest)     
    print("Train accuracy score: {:.8f}".format(curTrainAccuracy))
    print("done in %0.3fs." % (time()-startTime))

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
    #also remove all punctuation
    translator = str.maketrans('', '', string.punctuation)
    thread_df['Message Bodies'] = [s.translate(translator) for s in thread_df['Message Bodies']]
    #print (thread_df['Message Bodies'][1])
    
    #Find how many threads are marked solved
    num_solved = sum(thread_df['Solution Count']>0)
    num_threads = len(thread_df)
    percent_solved = (num_solved/num_threads)*100
    print(f"{num_solved} of {num_threads} threads are solved, or {percent_solved}%")
    
    #Create X and y data
    #IF USING TEST FILE: remove column ['Manual Solve']
    thread_X = thread_df.drop(columns=['Solution Count', 'Thread ID', 'Message List', 'User List', 'Message HTML', 'Post Times', 'Message Bodies'])
    #print (list(thread_X))
    #print (thread_X.head(n=5))
    thread_X_csr = csr_matrix(thread_X.values.astype(int))
    #print(thread_X_csr)
    thread_y = [False if x==0 else True for x in thread_df['Solution Count']]
    print('{} {}'.format(thread_y[0:10],thread_df['Solution Count'][0:10]))
    
    # Use tf-idf features
    print("Extracting tf-idf features...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       strip_accents = 'unicode',
                                       tokenizer = LemmaTokenizer(),
                                       ngram_range = (1,3),
                                       stop_words=all_stop_words_punct)
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(thread_df['Message Bodies'])
    print("done in %0.3fs." % (time() - t0))
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print (tfidf_feature_names)
    #print (type(tfidf))
    
    tfidf_thread_X = hstack((thread_X_csr, tfidf))
    #print (tfidf_thread_X)
    
    #Use tf (raw term count)
    print("Extracting tf features...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    strip_accents = 'unicode',
                                    tokenizer = LemmaTokenizer(),
                                    ngram_range = (1,3),
                                    stop_words=all_stop_words_punct)
    t0 = time()
    tf = tf_vectorizer.fit_transform(thread_df['Message Bodies'])
    print("done in %0.3fs." % (time() - t0))
    tf_feature_names = tf_vectorizer.get_feature_names()
    print (tf_feature_names)
    
    tf_thread_X = hstack((thread_X_csr, tf))
    
    print('Original dataset shape {}'.format(Counter(thread_y)))
    ncr = NeighbourhoodCleaningRule(random_state=42)
    tfidf_thread_X_res, tfidf_thread_y_res = ncr.fit_sample(tfidf_thread_X, thread_y)
    tf_thread_X_res, tf_thread_y_res = ncr.fit_sample(tf_thread_X, thread_y)
    thread_X_res, thread_y_res = ncr.fit_sample(thread_X, thread_y)
    print('Resampled dataset shape {}'.format(Counter(thread_y_res)))
    
    #try without text data
    print ('Creating model for no text training...')
    train_basic_rf(thread_X_res, thread_y_res)
    
    #Create 70-30 splits
    print ('Creating model for tfidf training...')
    train_basic_rf(tfidf_thread_X, tfidf_thread_y)
    
    #Create 70-30 splits
    print ('Creating model for tf training...')
    train_basic_rf(tf_thread_X_res, tf_thread_y_res)
    
if __name__ == '__main__': main()