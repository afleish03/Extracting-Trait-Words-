# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:33:37 2025

@author: arnol
"""

#create vector embeddings for our words
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

#define the model we're using to get vector embeddings
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

#define a function that computes cosine similarity of two vectors -- in other words,
#would tell us how similar two word embeddings are to each other 
def cosine_similarity(a, b): #input takes lists (vectors in list form)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


#get embeddings for the trait words. 
#sometimes distilbert breaks down words into subwords, so the loops at the end 
#of the function deal with this by averaging subword vectors into a single word vector
def obtain_sentence_embedding(sentence): #takes a string or list of strings as input 
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False, is_split_into_words = True)
    with torch.no_grad():
        outputs = model(**tokens)
    token_embeddings = outputs.last_hidden_state[0]  # Shape: [seq_len, hidden_size]
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])[1:-1]
    embeddings = []
    for i in range(len(decoded_tokens)): 
        if decoded_tokens[i] not in sentence: 
            for j in range(i,len(decoded_tokens)):
                for k in range(len(sentence)): 
                    if decoded_tokens[j][2:] == sentence[k][-len(decoded_tokens[j][2:]):]:
                        #average embedding vectors from i to j
                        average_embedding = np.average(np.array(token_embeddings)[1:-1][i:j+1], axis = 0).tolist()
                        #embeddings.append(this average)
                        embeddings.append(average_embedding)
                        break 
                break
        elif decoded_tokens[i] in sentence: 
            embeddings.append(token_embeddings[1:-1][i].tolist())
    return embeddings

#obtain word embeddings for the trait words + surrounding context 
contextualized_trait_embeddings = []
for i in range(len(flat_traits)):
    contextualized_trait_embeddings.append(obtain_sentence_embedding(flat_traits[i]))
    
#obtain the embedding of just the trait. This list of vectors will be used 
#to compare with words in the transcripts. 
trait_vectors = [0] * len(contextualized_trait_embeddings)
for i in range(len(contextualized_trait_embeddings)):
    trait_vectors[i] = contextualized_trait_embeddings[i][3]
    
#get trait words just to have then
words = [0] * len(flat_traits)
for i in range(len(words)): 
    words[i] = flat_traits[i][3]

    