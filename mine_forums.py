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

#define global variable for paths and data files
folder_path = get_my_file_path()

def splitListToRows(row,row_accumulator,target_column):
    split_row = ast.literal_eval(row[target_column])
    for s in split_row:
        new_row = row.to_dict()
        new_row[target_column] = s
        row_accumulator.append(new_row)
        

def main():
    #start by loading all data from the folder of txt files
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

    new_rows = []
    target_column = 'Message Bodies'
    message_df.apply(splitListToRows,axis=1,args = (new_rows,target_column))
    new_df = pd.DataFrame(new_rows)
    print (new_df.head(n=20))
        
if __name__ == '__main__': main()