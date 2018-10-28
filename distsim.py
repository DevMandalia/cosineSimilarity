from __future__ import division
import sys,json,math
import os
import numpy as np

def load_word2vec(filename):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.
    
    w2vec={}
    with open(filename,"r") as f_in:
        for line in f_in:
            line_split=line.replace("\n","").split()
            w=line_split[0]
            vec=np.array([float(x) for x in line_split[1:]])
            w2vec[w]=vec
    return w2vec

def load_contexts(filename):
    # Returns a dict containing a {word: contextcount} mapping.
    # It loads everything into memory.

    data = {}
    for word,ccdict in stream_contexts(filename):
        data[word] = ccdict
    print "file %s has contexts for %s words" % (filename, len(data))
    return data

def stream_contexts(filename):
    # Streams through (word, countextcount) pairs.
    # Does NOT load everything at once.
    # This is a Python generator, not a normal function.
    for line in open(filename):
        word, n, ccdict = line.split("\t")
        n = int(n)
        ccdict = json.loads(ccdict)
        yield word, ccdict

def cossim_sparse(v1,v2):
    # Take two context-count dictionaries as input
    # and return the cosine similarity between the two vectors.
    # Should return a number beween 0 and 1
    numerator_sum = 0
    denominator1_sum = 0
    denominator2_sum = 0
    cossim = 0
    for word1 in v1:
     if word1 in v2:
       count1 = v1.get(word1)  
       count2 = v2.get(word1)
       numerator_sum += (count1 * count2)

    for word1 in v1:
      count1 = v1.get(word1)
      denominator1_sum += (count1 * count1)

    for word2 in v2:
      count2 = v2.get(word2)
      denominator2_sum += (count2 * count2)
    
    cossim = numerator_sum / (math.sqrt(denominator1_sum) * math.sqrt(denominator2_sum))
    
    return cossim
    pass

def cossim_dense(v1,v2):
    # v1 and v2 are numpy arrays
    # Compute the cosine simlarity between them.
    # Should return a number between -1 and 1
    
    numerator_sum = np.sum(np.multiply(v1, v2))
    denominator1_sum = np.sqrt(np.sum(np.square(v1)))
    denominator2_sum = np.sqrt(np.sum(np.square(v2)))

    return np.divide(numerator_sum, np.multiply(denominator1_sum, denominator2_sum))
    pass

def show_nearest(word_2_vec, w_vec, exclude_w, sim_metric):
    #word_2_vec: a dictionary of word-context vectors. The vector could be a sparse (dictionary) or dense (numpy array).
    #w_vec: the context vector of a particular query word `w`. It could be a sparse vector (dictionary) or dense vector (numpy array).
    #exclude_w: the words you want to exclude in the responses. It is a set in python.
    #sim_metric: the similarity metric you want to use. It is a python function
    # which takes two word vectors as arguments.

    # return: an iterable (e.g. a list) of up to 10 tuples of the form (word, score) where the nth tuple indicates the nth most similar word to the input word and the similarity score of that word and the input word
    # if fewer than 10 words are available the function should return a shorter iterable
    #
    # example:
    #[(cat, 0.827517295965), (university, -0.190753135501)]
    similar_words = []
    score = 0
    
    for word1 in word_2_vec:
      if word1 not in exclude_w:
        if sim_metric.__name__ == "cossim_sparse":
          score = cossim_sparse(w_vec, word_2_vec.get(word1))
        if sim_metric.__name__ == "cossim_dense":
          score = cossim_dense(w_vec, word_2_vec.get(word1))
        this_tuple = (word1, score)
        similar_words.append(this_tuple)
    
    similar_words.sort(key=lambda elem: elem[1], reverse=True)
    if len(similar_words) > 10:
      similar_words = similar_words[0:10]
    
    return similar_words
    pass
    

