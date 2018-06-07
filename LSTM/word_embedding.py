import os
import pandas as pd
import numpy as np
import io
import string
import gensim
import ast

from nltk.tokenize import sent_tokenize
from nltk import word_tokenize


class SentenceIterator(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            print(self.dirname+fname)
            #Read in CSV
            with open(self.dirname+fname, 'r') as f:
                str1 = f.read()
            doc_df = pd.read_csv(io.StringIO(str1))

            #For each message body
            for w in doc_df['Message Bodies']:    
                message_list = ast.literal_eval(w)
                w = ' '.join(message_list)
                #Replace \xa0 and \n with ' '
                message_body = w.replace('\xa0', ' ').replace('\n', ' ').lower()
                #print(message_body)
                #Split into sentences
                st = sent_tokenize(message_body)
                st = [sent.replace(',',' ') for sent in st]
                #print(st)
                #Return each sentence tokenized
                for sentence in st:
                    
                    #Lemmitize here
                    yield word_tokenize(sentence)

#Word Embeddings
def do_word_embedding(full_file_path, verbose=False):

    sentences = SentenceIterator(full_file_path)

    if verbose:
        for i in sentences:
            print('---------------------')
            print(i)
    model = gensim.models.Word2Vec(sentences,size=200)

    return model
