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
    inumpyut:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag), (word2, tagnxt)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag_count = {}
    tags = [tag for sent in train for _, tag in sent]
    common_tag = max(set(tags), key = tags.count)       #get common tags of the whole dataset

    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag_count:
                word_tag_count[word] = {}
            if tag not in word_tag_count[word]:
                word_tag_count[word][tag] = 0
            word_tag_count[word][tag] += 1     #count the occurance of tags for each word
            
    word_tags = {}

    for word in word_tag_count:
        tags = word_tag_count[word]
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
# 	inumpyut:  training data (list of sentences, with tags on the words)
# 			test data (list of sentences, no tags on the words)
# 	output: list of sentences with tags on the words
# 			E.g., [[(word1, tag), (word2, tagnxt)], [(word3, tag3), (word4, tag4)]]
# 	'''
import math


def viterbi(train, test):
	'''
	Implementation for the viterbi tagger.
	inumpyut:  training data (list of sentences, with tags on the words)
			test data (list of sentences, no tags on the words)
	output: list of sentences with tags on the words
			E.g., [[(word1, tag), (word2, tagnxt)], [(word3, tag3), (word4, tag4)]]
	'''

	alpha = 1e-3
	
	tag_count = Counter()
	tag_pair_count = defaultdict(Counter)   #tag and next tag pair
	word_tag_count = defaultdict(Counter)
	tag_word_count = defaultdict(Counter)
	initial_p = defaultdict(float)
	transition_p = defaultdict(lambda: defaultdict(float))
	emission_p = defaultdict(lambda: defaultdict(float))
	set = [word_tag for sentence in train for word_tag in sentence]
	num_tag = len(set)
	
	for i in range(num_tag - 1):
		word, tag = set[i]
		next_tag = set[i+1][1]
		tag_count[tag] += 1
		tag_pair_count[tag][next_tag] += 1
		word_tag_count[word][tag] += 1
		tag_word_count[tag][word] += 1
		
	word, tag = set[-1]
	tag_count[tag] += 1
	word_tag_count[word][tag] += 1
	tag_word_count[tag][word] += 1

    #get three different probabilities
	for tag, num in tag_count.items():
		denom = alpha * (len(tag_count)) + tag_count['START']
		initial_likelihood = (tag_pair_count['START'][tag] + alpha) / denom
		initial_p[tag] = -np.log(initial_likelihood)

		for tagnxt in tag_count:
			count_tag_tagnxt = tag_pair_count[tag][tagnxt]
			count_tag = sum(tag_pair_count[tag].values())
			denom = count_tag + (len(tag_count)) * alpha
			transition_likelihood = (count_tag_tagnxt + alpha) / denom
			transition_p[tag][tagnxt] = -np.log(transition_likelihood)
			
		denom = tag_count[tag] + alpha * (len(tag_word_count[tag]) + 1)
		emission_p['UNKNOWN'][tag] = -np.log(alpha / denom)
		
		for word in tag_word_count[tag]:
			if tag in word_tag_count[word]:
				emission_likelihood = (word_tag_count[word][tag] + alpha) / denom
				emission_p[word][tag] = -np.log(emission_likelihood) 

	result = []
	for i, sentence in enumerate(test):
		result = viterbi_helper(sentence[1:], initial_p, transition_p, emission_p)
		test[i][0] = ("START", "START")
		
		for j, word in enumerate(sentence[1:]):
			test[i][j+1] = (word, result[j])

	return test

def viterbi_helper(words, initial_p, transition_p, emission_p):
	
	prob = defaultdict(lambda:defaultdict(float))
	backtrace = defaultdict(lambda:defaultdict())
	
	first = words[0]
	if first not in emission_p:
		first = 'UNKNOWN'
		
	for tag, pb in emission_p[first].items():
		prob[0][tag] = pb + initial_p[tag]

	for i, word in enumerate(words[1:]):
		idx = i + 1
		
		if word not in emission_p:
			word = 'UNKNOWN'
			
		for cur_tag in emission_p[word]:
			argmax_v = None
			max_v = math.inf
			
			for last in prob[idx-1]:
				current = prob[idx-1][last]+transition_p[last][cur_tag]+emission_p[word][cur_tag]
				if (current < max_v):
					max_v = current
					argmax_v = last
			prob[idx][cur_tag] = max_v
			backtrace[idx][cur_tag] = argmax_v

	final = None
	max = math.inf
	for tag in prob[len(words)-1]:
		current = prob[len(words)-1][tag]
		if (current < max):
			max = current
			final = tag

	result = [final]
	last = final
	
	for i in range(len(words) -1):
		idx = len(words) - 1 - i
		last = backtrace[idx][last]
		result.append(last)
		
		if (idx == 1):
			break
		
	result = [result[len(result)-i-1] for i in range(len(result))]

	return result	
	

def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    inumpyut:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag), (word2, tagnxt)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag), (word2, tagnxt)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



