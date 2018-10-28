#!/usr/bin/env python
import distsim
word_to_vec_dict = distsim.load_word2vec("nyt_word2vec.4k")
###Provide your answer below

###Answer examples; replace with your choices

print("Word 1 is::::america")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_vec_dict, word_to_vec_dict['america'],set(['america']),distsim.cossim_dense)):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 2 is::::years")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_vec_dict, word_to_vec_dict['years'],set(['years']),distsim.cossim_dense)):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 3 is::::great")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_vec_dict, word_to_vec_dict['great'],set(['great']),distsim.cossim_dense)):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 4 is::::run")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_vec_dict, word_to_vec_dict['run'],set(['run']),distsim.cossim_dense)):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 5 is::::between")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_vec_dict, word_to_vec_dict['between'],set(['between']),distsim.cossim_dense)):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 6 is::::wife")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_vec_dict, word_to_vec_dict['wife'],set(['wife']),distsim.cossim_dense)):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")
