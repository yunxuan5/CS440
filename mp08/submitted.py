'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np
import utils

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag_counts = {}
    tags = [tag for sent in train for _, tag in sent]
    common_tag = max(set(tags), key = tags.count)       #get common tags of the whole dataset

    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag_counts:
                word_tag_counts[word] = {}
            if tag not in word_tag_counts[word]:
                word_tag_counts[word][tag] = 0
            word_tag_counts[word][tag] += 1     #count the occurance of tags for each word
            
    word_tags = {}

    for word in word_tag_counts:
        tags = word_tag_counts[word]
        most_common_tag = max(tags, key=tags.get)
        word_tags[word] = most_common_tag       #get the most common tag of each word
    
    # apply the dictionary to the test data to assign a tag to each word
    tagged_sentences = []
    for sentence in test:
        tagged_word = []
        for word in sentence:
            if word in word_tags:
                tag = word_tags[word]   # assign most common tag to each words
            else:
                tag = common_tag  # assign common tag to unseen words
            tagged_word.append((word, tag))

        tagged_sentences.append(tagged_word)

    return tagged_sentences
#     raise NotImplementedError("You need to write this part!")


# def viterbi(train, test):
# 	'''
# 	Implementation for the viterbi tagger.
# 	input:  training data (list of sentences, with tags on the words)
# 			test data (list of sentences, no tags on the words)
# 	output: list of sentences with tags on the words
# 			E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
# 	'''
import math

def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
            alpha: smoothing constant (default=1)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    train_set = [word_tag for sentence in train for word_tag in sentence]
    alpha = 1e-3
    # Count occurrences of tags, tag pairs, tag/word pairs
    tag_count = {} # count of each tag
    tag_pair_count = {} # count of each tag pair
    tag_word_count = {} # count of each tag/word pair
    word_set = set() # set of all words in the training data
    
    for sentence in train:
        for i in range(len(sentence)):
            word, tag = sentence[i]
            if tag not in tag_count:
                tag_count[tag] = 1
            else:
                tag_count[tag] += 1
            if i == 0:
                # First word in sentence
                if ("START", tag) not in tag_pair_count:
                    tag_pair_count[("START", tag)] = 1
                else:
                    tag_pair_count[("START", tag)] += 1
            else:
                # Not first word in sentence
                prev_tag = sentence[i-1][1]
                if (prev_tag, tag) not in tag_pair_count:
                    tag_pair_count[(prev_tag, tag)] = 1
                else:
                    tag_pair_count[(prev_tag, tag)] += 1
            if (tag, word) not in tag_word_count:
                tag_word_count[(tag, word)] = 1
            else:
                tag_word_count[(tag, word)] += 1
            word_set.add(word)
    
	# for i in range(len(train_set)-1):
    #      word, tag = train_set[i]
    #      next_tag = train_set[i+1][1]
    #      tag_count[tag] += 1
    #      tag_pair_count[tag][next_tag] += 1
    #      word
        
    
    # Compute smoothed probabilities
    num_tags = len(tag_count)
    initial_prob = {}
    transition_prob = {}
    emission_prob = {}
    
    for tag in tag_count:
        # Compute initial probability
        initial_prob[tag] = (tag_pair_count[("START", tag)] + alpha) / (tag_count["START"] + alpha * num_tags)
        # Compute transition probabilities
        for next_tag in tag_count:
            transition_prob[(tag, next_tag)] = (tag_pair_count.get((tag, next_tag), 0) + alpha) / (tag_count[tag] + alpha * num_tags)
        # Compute emission probabilities
        for word in word_set:
            emission_prob[(tag, word)] = (tag_word_count.get((tag, word), 0) + alpha) / (tag_count[tag] + alpha*(len(word_set)+1))
        # Compute emission probability for "UNKNOWN" words
        emission_prob[(tag, "UNKNOWN")] = alpha / (tag_count[tag] + alpha*(len(word_set)+1))
    
    # Take the log of each probability
    for key in initial_prob:
        initial_prob[key] = math.log(initial_prob[key])
    for key in transition_prob:
        transition_prob[key] = math.log(transition_prob[key])
    for key in emission_prob:
        emission_prob[key] = math.log(emission_prob[key])
        
	# result = []
    result = []
    for i, sentence in enumerate(test):
         try:
              result = _viterbi(sentence[1:], initial_prob, transition_prob, emission_prob)
         except:
             print(i)
         test[i][0] = ("START", "START")
         for j, word in enumerate(sentence[1:]):
             try:
                  test[i][j+1] = (word, result[j])
             except:
                  print(i, j)
    return test

    
    # Construct the trellis
	# result = []
	# for j, sentence in enumerate(test):
              
        
    # for j, sentence in enumerate(test):
	#     try:
	# 	    result = _viterbi(sentence[1:], initial_prob, transition_prob, emission_prob)
    #     except:
	#         print(j)
	#     test[j][0] = ("START", "START")
	# 	# test[j][-1] = ('END', 'END')
	#     for i, word in enumerate(sentence[1:]):
	# 		try:
	# 			test[j][i+1] = (word, result[i])
	# 		except:
	# 			print(j, i)

	# return test


# 	train_set = [word_tag for sentence in train for word_tag in sentence]
	
# 	## Count occurrences of tags, tag pairs, and tag/word pairs
# 	tag_counts = Counter()
# 	tag_pair_counts = defaultdict(Counter)
# 	word_tag_counts = defaultdict(Counter)
# 	tag_word_counts = defaultdict(Counter)
	
# 	for i in range(len(train_set)-1):
# 		word, tag = train_set[i]
# 		# if tag == "START" or tag == "END":
# 		# 	continue
# 		next_tag = train_set[i+1][1]
# 		tag_counts[tag] += 1
# 		tag_pair_counts[tag][next_tag] += 1
# 		word_tag_counts[word][tag] += 1
# 		tag_word_counts[tag][word] += 1
# 	word, tag = train_set[-1]
# 	tag_counts[tag] += 1
# 	word_tag_counts[word][tag] += 1
# 	tag_word_counts[tag][word] += 1

    # num_tags = len(tag_count)
    # initial_prob = {}
    # transition_prob = {}
    # emission_prob = {}
    # for tag in tag_count:
    #     # Compute initial probability
    #     initial_prob[tag] = (tag_pair_count[("START", tag)] + alpha) / (tag_count[tag] + alpha*num_tags)
    #     # Compute transition probabilities
    #     for next_tag in tag_count:
    #         transition_prob[(tag, next_tag)] = (tag_pair_count.get((tag, next_tag), 0) + alpha) / (tag_count[tag] + alpha*num_tags)
    #     # Compute emission probabilities
    #     for word in word_set:
    #         emission_prob[(tag, word)] = (tag_word_count.get((tag, word), 0) + alpha) / (tag_count[tag] + alpha*(len(word_set)+1))
    #     # Compute emission probability for "UNKNOWN" words
    #     emission_prob[(tag, "UNKNOWN")] = alpha / (tag_count[tag] + alpha*(len(word_set)+1))
    
    # # Take the log of each probability
    # for key in initial_prob:
    #     initial_prob[key] = math.log(initial_prob[key])
    # for key in transition_prob:
    #     transition_prob[key] = math.log(transition_prob[key])
    # for key in emission_prob:
    #     emission_prob[key] = math.log(emission_prob[key])
    
# 	# Construct the trellis
# 	result = []
# 	for j, sentence in enumerate(test):
# 		try:
# 			result = _viterbi(sentence[1:], initial_prob, transition_prob, observation_prob)
# 		except:
# 			print(j)
# 		test[j][0] = ("START", "START")
# 		# test[j][-1] = ('END', 'END')
# 		for i, word in enumerate(sentence[1:]):
# 			try:
# 				test[j][i+1] = (word, result[i])
# 			except:
# 				print(j, i)

# 	return test

def _viterbi(words, initial_prob, transition_prob, observation_prob):
	## Initialization
	node_prob = defaultdict(lambda:defaultdict(float))
	backtrace = defaultdict(lambda:defaultdict())
	
	first_word = words[0]
	if first_word not in observation_prob:
		first_word = 'UNKNOWN'
	for tag, prob in observation_prob[first_word].items():
		node_prob[0][tag] = prob + initial_prob[tag]
	
	## Iteration
	for i, word in enumerate(words[1:]):
		index = i + 1
		if word not in observation_prob:
			word = 'UNKNOWN'
		for cur_tag in observation_prob[word]:
			arg_max_v = None
			max_v = float('inf')
			for last_tag in node_prob[index-1]:
				cur_val = node_prob[index-1][last_tag]+transition_prob[last_tag][cur_tag]+observation_prob[word][cur_tag]
				if (cur_val < max_v):
					max_v = cur_val
					arg_max_v = last_tag
			node_prob[index][cur_tag] = max_v
			backtrace[index][cur_tag] = arg_max_v

	## Termination
	final_tag = None
	max_prob = float('inf')
	for tag in node_prob[len(words)-1]:
		cur_val = node_prob[len(words)-1][tag]
		if (cur_val < max_prob):
			max_prob = cur_val
			final_tag = tag

	## Back-Trace
	result_tag = [final_tag]
	last_tag = final_tag
	for i in range(len(words) -1):
		index = len(words) - 1 - i
		last_tag = backtrace[index][last_tag]
		result_tag.append(last_tag)
		if (index == 1):
			break
		
	result_tag = [result_tag[len(result_tag)-i-1] for i in range(len(result_tag))]

	return result_tag	
	

def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



