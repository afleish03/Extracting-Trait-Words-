# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:35:28 2025

@author: arnol
"""

from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import matplotlib.pyplot as plt 

# os.chdir('C:\\Users\\arnol\\OneDrive\\Desktop\\CS projects')
# from train_data_processor import simple_traits 

#define model 
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

#define a function that computes cosine similarity of two vectors -- used to 
#compare whether two embeddings represent similar meanings 
def cosine_similarity(a, b): #input lists 
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

#obtain embedding for each word of these transcripts. Each word becomes a 700+-dimensional
#vector, which is very computationally intensive 
def vectorize_tokenized_transcripts(transcripts): #list of lists of strings which are transcripts
    vectorized_transcripts = []
    decoded_transcripts = []
    for transcript in transcripts: 
        if transcript != []:
            tokens = tokenizer(transcript, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False, is_split_into_words = True)
            with torch.no_grad():
                outputs = model(**tokens)
            token_embedding = outputs.last_hidden_state[0].tolist()
            decoded_token = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
            vectorized_transcripts.append(token_embedding)
            decoded_transcripts.append(decoded_token)
        else: 
            vectorized_transcripts.append(np.zeros(768).tolist())
    return vectorized_transcripts

#returns vector representation of tokens used in each trial 
test_data_embeddings = vectorize_tokenized_transcripts(processed_test_transcripts)

#flatten out all of the tokens used
flattened_test_embeddings = [vector for transcript in test_data_embeddings for vector in transcript]

#for each word used in all transcripts, check how similar that word is to 
#trait embeddings, on average 
avg_cos_sims = []
for embedding in flattened_test_embeddings: 
    cos_sims = []
    for trait in trait_vectors: 
        cos_sims.append(cosine_similarity(embedding, trait))
    avg_cos_sims.append(np.average(cos_sims,axis = 0))