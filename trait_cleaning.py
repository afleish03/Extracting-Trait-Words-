# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:46:49 2025

@author: arnol
"""

import pandas as pd 
from spellchecker import SpellChecker
from nltk import word_tokenize
import contractions




#trait word data is messy and requires some manual adjustments

#assumes traits (and all other trial data) is stored in a csv called TranscriptData 
df = pd.read_csv('TranscriptData.csv') 
#drop unnecessary columns in data, truncate empty cells 
df.drop((df.columns[i] for i in [0,1,3,4,5,8]), axis = 1, inplace = True).truncate(after = 619)
df.dropna(axis = 'rows', subset =['RAW TEXT'], inplace = True) #drop nans 
df.reset_index(inplace = True)

#extract traits from the dataset. assumed traits are stored in the column called 
#'list of trait words'
list_of_traits = df['List of trait words'].tolist()





#define a spellchecker and add some used words to the dictionary 
spell = SpellChecker()
spell.word_frequency.load_words(['mansplainer', 'fuckboy', 'skeevy', 'scammer', 'hardworker'])

def process_words(in_list): 
    corrected_phrases = []
    for i in range(len(in_list)):
        corrected_sentence = []
        if type(in_list[i]) == str: 
            in_list[i] = in_list[i].split(",")
            #fix contractions that may have been used 
            in_list[i] = [contractions.fix(trait).strip().lower() for trait in in_list[i]]
            in_list[i] = [trait.replace('hard worker', 'hardworker').replace('hard working', 'hardworking') for trait in in_list[i]] #fix consequence of transcription
            #run a spellchecker on the trait words 
            for j in range(len(in_list[i])): 
                corrected_words = " ".join([spell.correction(word) for word in in_list[i][j].split()])
                corrected_sentence.append(corrected_words)
            corrected_phrases.append(corrected_sentence)
        else: 
            corrected_phrases.append(None)
    return corrected_phrases

#store corrected trait words in a new column of data 
corrected_traits = process_words(list_of_traits)

#trial 182 and 184 indexed incorrectly in data, trait words are swapped. 
corrected_traits[184] = corrected_traits[182]
corrected_traits[182] = None


#now go through and parse where trait word = text to obtain surrounding context 
#of that trait word 
def obtain_traits_context(traits, transcripts, n):
    d = []
    for i in range(len(traits)): 
        if type(traits[i]) == list: 
            c = []
            for j in range(len(traits[i])): 
                b = []
                for k in range(len(transcripts[i])): 
                    if traits[i][j] == transcripts[i][k]:
                        a = []
                        if k - n < 0: 
                            for l in range(0,k+n): 
                                a.append(transcripts[i][l])
                        elif k + n > len(transcripts[i])- 1: 
                            for l in range(k-n, len(transcripts[i])): 
                                a.append(transcripts[i][l])
                        else: 
                            for l in range(k-n,k+n):
                                a.append(transcripts[i][l])
                        while a.index(traits[i][j]) < n: 
                            a.insert(0,'<PAD>')
                        b.append(a)
                c.append(b)    
            d.append(c)
        else:
            d.append(None)
    return d

contextualized_traits = obtain_traits_context(corrected_traits, processed_transcripts, 3)

#needed to perform a series of ugly manual adjustments to the data 

#manually correct -- 34, 116, 136, 191, 229, 385, 400, 401, 410, 411, 470, 479, 507, 605
contextualized_traits[34][0].pop() #for using flagged word not as trait word 
contextualized_traits[34][0].pop()
contextualized_traits[116][0][1] =contextualized_traits[116][0][1][3:] #for length 
contextualized_traits[136][0].pop(0) #for using flagged word not as trait word
contextualized_traits[191][0][1] = contextualized_traits[191][0][1][2:] #for length
contextualized_traits[229][0].pop(4) #for consistency, and for not using flagged word as trait word
contextualized_traits[229][0].pop(2)
contextualized_traits[229][0].pop(1)
contextualized_traits[385][0][1] = contextualized_traits[385][0][1][2:] #for length
contextualized_traits[400][0][1]= contextualized_traits[400][0][1][3:] #for length
contextualized_traits[401][0][1]= contextualized_traits[401][0][1][3:]
contextualized_traits[410][1][1] = contextualized_traits[410][1][1][3:]
contextualized_traits[411][0][1] = contextualized_traits[411][0][1][3:]
contextualized_traits[470][0][1] = contextualized_traits[470][0][1][3:]
contextualized_traits[471][0][1] = contextualized_traits[471][0][1][2:]
contextualized_traits[479][1].pop() #for simplicity
contextualized_traits[507][0][1] = contextualized_traits[507][0][1][3:]
contextualized_traits[605][0].pop() #for using word not as trait 

#flatten the list of contexualized traits, so that we have just a list of trait words used 
#in their context 
contextualized_traits = [item for item in contextualized_traits if item is not None]        
contextualized_traits = [item for sublist in contextualized_traits for item in sublist]
flat_traits = [item for sublist in contextualized_traits for item in sublist]

#extra adjustments missed beforehand 
flat_traits[109] = flat_traits[109][3:]
flat_traits[111] = flat_traits[111][1:]
flat_traits[113] = flat_traits[113][2:]

#flat_traits leaves us with a list of traits and their contexts, which 
#we can now use to get context-based embeddings for our traits 
