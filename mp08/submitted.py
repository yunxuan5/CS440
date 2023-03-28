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


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    print(test)
    tag_counts = {}
    transition_counts = {}
    emission_counts = {}
    for sentence in train:
        previous_tag = "START"
        for word, tag in sentence:
            if tag not in tag_counts:
                tag_counts[tag] = 0
            tag_counts[tag] += 1
            if previous_tag not in transition_counts:
                transition_counts[previous_tag] = {}
            if tag not in transition_counts[previous_tag]:
                transition_counts[previous_tag][tag] = 0
            transition_counts[previous_tag][tag] += 1
            if tag not in emission_counts:
                emission_counts[tag] = {}
            if word not in emission_counts[tag]:
                emission_counts[tag][word] = 0
            emission_counts[tag][word] += 1
            previous_tag = tag
    
    # Compute the probabilities from the counts
    tag_probabilities = {}
    for tag in tag_counts:
        tag_probabilities[tag] = tag_counts[tag] / sum(tag_counts.values())
    transition_probabilities = {}
    for previous_tag in transition_counts:
        transition_probabilities[previous_tag] = {}
        for tag in transition_counts[previous_tag]:
            transition_probabilities[previous_tag][tag] = transition_counts[previous_tag][tag] / sum(transition_counts[previous_tag].values())
    emission_probabilities = {}
    for tag in emission_counts:
        emission_probabilities[tag] = {}
        for word in emission_counts[tag]:
            emission_probabilities[tag][word] = emission_counts[tag][word] / sum(emission_counts[tag].values())
    
    # Apply the Viterbi algorithm to each sentence in the test data
    tagged_sentences = []
    for sentence in test:
        # Initialize the Viterbi tables
        viterbi = [{tag: tag_probabilities[tag] * emission_probabilities[tag].get(sentence[0], 0) for tag in tag_probabilities}]
        backpointer = [{}]
        # Iterate over the rest of the words in the sentence
        for word in sentence[1:]:
            viterbi.append({})
            backpointer.append({})
            for tag in tag_probabilities:
                max_score = 0
                max_previous_tag = None
                for previous_tag in viterbi[-2]:
                    score = viterbi[-2][previous_tag] * transition_probabilities[previous_tag].get(tag, 0) * emission_probabilities[tag].get(word, 0)
                    if score > max_score:
                        max_score = score
                        max_previous_tag = previous_tag
                viterbi[-1][tag] = max_score
                backpointer[-1][tag] = max_previous_tag
        # Find the highest-scoring tag sequence for the sentence
       


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



