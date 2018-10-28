#!/usr/bin/env python
import distsim

word_to_ccdict = distsim.load_contexts("nytcounts.4k")

### provide your answer below
# proper-noun america
# common-noun years
# adjective great
# verb run
# word 5 between
# word 6 wife

print("Word 1 is::::america")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_ccdict, word_to_ccdict['america'],set(['america']),distsim.cossim_sparse), start=1):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 2 is::::years")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_ccdict, word_to_ccdict['years'],set(['years']),distsim.cossim_sparse), start=1):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 3 is::::great")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_ccdict, word_to_ccdict['great'],set(['great']),distsim.cossim_sparse), start=1):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 4 is::::run")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_ccdict, word_to_ccdict['run'],set(['run']),distsim.cossim_sparse), start=1):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 5 is::::between")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_ccdict, word_to_ccdict['between'],set(['between']),distsim.cossim_sparse), start=1):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Word 6 is::::wife")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_ccdict, word_to_ccdict['wife'],set(['wife']),distsim.cossim_sparse), start=1):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")

print("Further exploring:big from great")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_ccdict, word_to_ccdict['big'],set(['big']),distsim.cossim_sparse), start=1):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")


print("Further exploring:tiny from great")
for i, (word, score) in enumerate(distsim.show_nearest(word_to_ccdict, word_to_ccdict['tiny'],set(['tiny']),distsim.cossim_sparse), start=1):
    print("{}: {} ({})".format(i, word, score))
print("--------------------")