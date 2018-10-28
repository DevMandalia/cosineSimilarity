#!/usr/bin/env python
import distsim
from collections import defaultdict


word_to_vec_dict = distsim.load_word2vec("nyt_word2vec.4k")
task_accuracies = defaultdict(list)

file = open("word-test.v3.txt", "r")

for line in file:
  line = line.strip('\n')
  if line.startswith('//'):
    continue
  if line.startswith(":"):
    category = line[2:]
  else:
    words = line.split()
    w1_dict = word_to_vec_dict[words[0]]
    w2_dict = word_to_vec_dict[words[1]]
    w4_dict = word_to_vec_dict[words[3]]
    result = distsim.show_nearest(word_to_vec_dict, w1_dict - w2_dict + w4_dict, set([words[0],words[1],words[3]]), distsim.cossim_dense)
    i = 0
    match = False
    for vec in result:
      i+=1
      if vec[0] == words[2]:
        match = True
        break
    if not match:
      i = 0 
    task_accuracies[category].append(i)

for key, value in task_accuracies.items():
  top1 = 0
  top5 = 0
  top10 = 0
  for position in value:
    if position is not 0:
      if position == 1: top1 += 1
      if position <= 5: top5 += 1
      if position <= 10: top10 += 1
  print(key + ":     " + str(round(float(top1)/float(len(value)), 2)) + "    " +
                       str(round(float(top5)/float(len(value)), 2)) + "    " + str(round(float(top10)/float(len(value)), 2)))

    

    
