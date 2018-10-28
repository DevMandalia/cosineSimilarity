#  Cosine Similarity


1) cossim sparse(v1,v2) in distsim.py computes and displays the cosine similarities between each pair of the words 
in nytcounts.university_cat_dog
testq1.py tests its running correctly

2) Given a dictionary of word-context vectors, the context vector of a particular
query word w, the words you want to exclude in the responses (the query word w in this
question), and the similarity metric you want to use (it should be the cossim sparse function you just
implemented), show_nearest() finds the 10 words most similar to w.  testq2.py tests its running correctly

3) pick 6 words from file vocab and show the top 10 most similar words by running the show_nearest() method on each of them.
q3.py runs show_nearest() on 6 words picked from vocab

4) cossim dense(v1,v2) in distsim.py computes and displays the cosine similarities between each pair of words from 
nyt_word2vec.university_cat_dog which contains word embedding vectors pretrained by word2vec. Inputs of 
cossim_dense(v1,v2) are numpy arrays.

5) q5.py is repeat of (3) but using cossim_dense

6) testq6.py performs analogical reasoning tasks using word vectors abd gives the answer for => king : man :: __ : woman

7) q7.py tries out the word2vec vectors out on eight different analogy tasks using the file: word-test.v3.txt. The
groups of relations are delimited by lines starting with a colon (:) and we see these groups: capital,
currency, city-in-state, family, adjective-to-adverb, comparative, superlative, and nationality-adjective.q7.py prints the 
 1-best, 5-best, and 10-best accuracy of the vectors on ach of the eight relation groups.
