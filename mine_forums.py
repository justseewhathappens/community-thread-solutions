#libraries
import re
import json
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import io

#define global variable for paths and data files
folder_path = 'C:/Users/jwolfgan/Documents/Personal/UT/4b - Machine Learning/Final Project/Data/data_output_cleaned/'

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
        print(doc_df.head())
        thread_df = thread_df.append(doc_df, ignore_index= True)
        
    print ("Final threads df:")
    print (list(thread_df))
    #print (thread_df.head(n=5))
    print (thread_df['Message HTML'][1])
        
if __name__ == '__main__': main()