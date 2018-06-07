import os
import io
import pandas as pd
import ast
from nltk import word_tokenize

def get_data(dirname):
    thread_df = pd.DataFrame()
    
    for fname in os.listdir(dirname):
        print(dirname+fname)        
        with open(dirname+fname, 'r') as f:
            str1 = f.read()
        doc_df = pd.read_csv(io.StringIO(str1))
        #print(type(doc_df['Message Bodies'][0]))
        #print(doc_df['Message Bodies'][0])
        #print(doc_df['Message Bodies'][1])
        thread_df = thread_df.append(doc_df, ignore_index= True)

    return thread_df

def get_longest_message(df, key):
    longest = 0
    for row in df[key]:
        l = len(row)
        #print(l)
        if(l>longest):
            longest = l
    return longest        

def batch_data (dirname):

    thread_df = get_data(dirname)
    #Standard Remove Stuff and to loer case
    thread_df['Message Bodies'] = [ast.literal_eval(w) for w in thread_df['Message Bodies']]

    new_column = []
    for row in thread_df['Message Bodies']:
        new_row = []
        for line in row:
            #Lemmatize here on this silly line - Break it up
            new_row.extend(word_tokenize(line.replace('\\xa0', ' ').replace('\\n', ' ').replace(',', ' ').lower()))
        new_column.append(new_row)    
    thread_df['Word Bodies'] = new_column
    
    #thread_df['Message Bodies'] = [w.replace('\\xa0', ' ').replace('\\n', ' ').lower() for w in thread_df['Message Bodies']]
    #thread_df['Message Bodies'] = [word_tokenize(w) for w in thread_df['Message Bodies']]
    
    #print(thread_df['Word Bodies'][0])
    
    num_threads = len(thread_df['Word Bodies']);
    largest_size = get_longest_message(thread_df, 'Word Bodies')

    return thread_df, num_threads, largest_size
    
#batch_data( 'C:\\Users\\Jonathan\\Documents\\Python\\Final_Project\\Data\\')  

#batch_data('C:\\Users\\Jonathan\\Documents\\Python\\Final_Project\\Data\\')
