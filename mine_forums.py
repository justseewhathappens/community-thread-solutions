#libraries
import re
import json
import os
import time
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

#define global variable for paths and data files
folder_path = get_my_file_path()
t = MosesTokenizer()
stop_words = set(stopwords.words('english'))
more_stop_words = [',','&quot;','(',')','/','&apos;t','&apos;re','&apos;s','&apos;ve','&gt;','+','~','-','*','\\',':','--', '\'',
                   '#','$','%','&amp;','&apos;','&apos;d','&apos;ll','&apos;m','..','...','....','"']
punct_stop_words = ['?','.']
all_stop_words = stop_words.union(more_stop_words)
all_stop_words_punct = all_stop_words.union(punct_stop_words)

def splitListToRows(row,row_accumulator,target_column):
    split_row = ast.literal_eval(row[target_column])
    for s in split_row:
        s = unidecode(s)
        new_row = row.to_dict()
        new_row[target_column] = s
        row_accumulator.append(new_row)

#tokenize a message and remove stop words
def splitTokensToRows(row,row_accumulator,target_column):
    split_row = t.tokenize(row[target_column])
    for s in split_row:
        s_lower = s.lower()
        if not s_lower in all_stop_words:
            new_row = row.to_dict()
            new_row[target_column] = s_lower
            row_accumulator.append(new_row)
            
#tokenize a message to bigrams and remove stop words
def splitBigramsToRows(row,row_accumulator,target_column):
    split_row = bigrams(t.tokenize(row[target_column]))
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
    #print (thread_df.head(n=5))
    #print (thread_df['Message HTML'][1])
    
    #Find how many threads are marked solved
    num_solved = sum(thread_df['Solution Count']>0)
    num_threads = len(thread_df)
    percent_solved = (num_solved/num_threads)*100
    print(f"{num_solved} of {num_threads} threads are solved, or {percent_solved}%")
    
    #create a dataframe of the messages, for tokenizing
    message_df = thread_df[['Thread ID','Message Bodies']]
    print (list(message_df))
    print (message_df.head(n=20))
    print (type(message_df['Message Bodies'][0]))

    #split the list of messages into rows
    new_rows = []
    target_column = 'Message Bodies'
    message_df.apply(splitListToRows,axis=1,args = (new_rows,target_column))
    new_msg_df = pd.DataFrame(new_rows)
    print (new_msg_df['Message Bodies'][0])
    
    ###TOKENS###
    #split the messages into lists of tokens - target column is still message bodies
    new_rows = []  #reinitialize
    new_msg_df.apply(splitTokensToRows,axis=1,args = (new_rows,target_column))
    token_df = pd.DataFrame(new_rows)
    print (token_df.head(n=20))
       
    #get count of all tokens and see most common 
    token_counts = token_df['Message Bodies'].value_counts()
    num_unique_tokens = len(token_counts)
    print(f'num tokens: {num_unique_tokens}')
    print(token_counts[0:20])
    
    #see count of tokens per thread
    #this will output a 3 column series that needs to be converted to a df and unstacked
    token_counts_thread = token_df['Message Bodies'].groupby(token_df['Thread ID']).value_counts()
    #print ("\n\nAfter first calculation...")
    #print(token_counts_thread[0:20])
    token_counts_thread_df =pd.DataFrame(token_counts_thread)
    #print ("\n\nAfter conversion to DF...")
    #print (token_counts_thread_df.head(n=20))
    token_counts_thread_df = token_counts_thread_df.unstack(fill_value=0)
    print ("\n\nAfter unstacking...")
    print (token_counts_thread_df.head(n=20))

    
    ###BIGRAMS###
    #split the messages into lists of bigrams - target column is still message bodies
    new_rows = []  #reinitialize
    new_msg_df.apply(splitBigramsToRows,axis=1,args = (new_rows,target_column))
    bigram_df = pd.DataFrame(new_rows)
    print (bigram_df.head(n=20))
       
    #get count of all tokens and see most common 
    bigram_counts = bigram_df['Message Bodies'].value_counts()
    num_unique_bigrams = len(bigram_counts)
    print(f'num bigrams: {num_unique_bigrams}')
    print(bigram_counts[0:20])
    
    #see count of bigrams per thread
    #this will output a 3 column series that needs to be converted to a df and unstacked
    bigram_counts_thread = bigram_df['Message Bodies'].groupby(bigram_df['Thread ID']).value_counts()
    bigram_counts_thread_df =pd.DataFrame(bigram_counts_thread).unstack(fill_value=0)
    print ("\n\nAfter unstacking bigrams...")
    print (bigram_counts_thread_df.head(n=20))
    
if __name__ == '__main__': main()