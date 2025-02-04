# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:54:28 2025

@author: arnol
"""

import pandas as pd 
import os 
import contractions
import re
import num2words
from spellchecker import SpellChecker
from nltk.corpus import stopwords

#assumes transcripts (and all other trial data) is stored in a csv called TranscriptData 
df = pd.read_csv('TranscriptData.csv') 
#drop unnecessary columns in data, truncate empty cells 
df.drop((df.columns[i] for i in [0,1,3,4,5,8]), axis = 1, inplace = True).truncate(after = 619)
df.dropna(axis = 'rows', subset =['RAW TEXT'], inplace = True) #drop nans 
df.reset_index(inplace = True)

#define our set of stopwords. combine a pre-defined set of stopwords with custom words
#that appear frequently in transcription, and remove words we might wnat to hold on to
stop_words = set(stopwords.words('english'))
words_to_keep = ['he', 'his', 'him', 'she', 'her', 'more', 'is'] #words to 
words_to_remove = {'okay', 'alright', 'whoa'}
for i in range(len(words_to_keep)):
    stop_words.remove(words_to_keep[i])
for j in range(len(words_to_remove)):
    stop_words.update(words_to_remove)


#define a function that cleans up all of the transcript text 
def process_raw_text(input_col): #takes a list of transcripts from the data as input 
    if type(input_col) != list:
        x = input_col.tolist()
    else: 
        x = input_col
        
    #these are common words/phrases that appear in transcription but that we want to remove
    pattern = r'\b(u+m+|u+m+h+|m+h+m+|m+h+|h+m+|m+|u+h+|h+u+h+|a+h+|o+h+|h+o+|o+k+|aw+|o+)\b' 
    for i in range(len(x)):
        if type(x[i]) != float: 
            #remove very basic consequences of transcription 
            x[i] = re.sub(pattern, '', x[i], flags=re.IGNORECASE) #remove common words/phrases that appear in transcription
            #remove some other consequences of transciption
            x[i] = x[i].replace('<affirmative>', '').replace('<laugh>','').replace('<unintelligible>','').replace('$',' dollar ').replace('indiscernible', '').replace('indistinguishable','').replace('hard working','hardworking').replace('hard worker', 'hardworker')  
            x[i] = contractions.fix(x[i].strip().lower()) #fix contractions, lowercase, strip leading and trailing spaces 
            #process different ways of giving affirmative statements. may not be necessary depending on 
            #what tools are used to obtain semantic meaning of the words. 
            x[i] = x[i].replace('yeah', 'yes').replace('yea', 'yes').replace('yep', 'yes') 

            
            #numerical processing -- remove common bugs in number representation 
            x[i] = re.sub(r'(?<=\d)[^\w\s](?!\d)', ' ', x[i]) #remove periods but not decimal points for number comprehension
            #convert numbers to words:
            x[i] = ' '.join([num2words.num2words(j) if bool(re.match(r"^-?\d+(\.\d+)?$", j)) else j for j in x[i].split()]) 
            
            #remove the rest of punctuation, clean up spaces, remove stopwords
            x[i] = re.sub(r'[^\w\s]', ' ', x[i]) #remove all punctuation (after fixing contractions)
            x[i] = re.sub(r'\s{2,}', ' ', x[i])  # Replace two or more spaces with one
            x[i] = ' '.join([word for word in x[i].split() if word.lower() not in stop_words]) #remove stopwords -- also fixes some spelling issues
            #remove straggler numbers that might be left over due to some oversight 
            x[i] = ' '.join([num2words.num2words(j) if bool(re.match(r"^-?\d+(\.\d+)?$", j)) else j for j in x[i].split()]) 

            
            #can run spellchecker after all numbers are gone if necessary, but 
            #can often make things worse if speakers use slang, uncommon words, or names 
            # x[i] = correct_spelling(x[i])          
            
    return(x)


processed_transcripts = process_raw_text(df['RAW TEXT'])  
#assumes transcripts are stored in a column called 'raw text' 

