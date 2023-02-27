'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    # raise RuntimeError("You need to write this part!")
    cnt_pos = Counter()
    cnt_neg = Counter()
    for word in train['pos']:
        cnt_pos+=Counter(word)
    for word in train['neg']:
        cnt_neg+=Counter(word)
    frequency = {'pos':cnt_pos, 'neg':cnt_neg}
    return frequency
    

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    # raise RuntimeError("You need to write this part!")
    import copy
    nonstop = copy.deepcopy(frequency)
    # print(stopwords)
    # print(frequency['pos'])
    for word in nonstop['pos'].copy():
        if word in stopwords:
            del nonstop['pos'][word]
    for word in nonstop['neg'].copy():
        if word in stopwords:
            del nonstop['neg'][word]
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    # raise RuntimeError("You need to write this part!")

    likelihood = {'pos':{}, 'neg':{}}
    for word in nonstop['pos'].copy():
        sum_pos = sum(nonstop['pos'].values())
        pos_denom = sum_pos + smoothness*(len(set(nonstop['pos'])) + 1)
    for word in nonstop['neg'].copy():
        sum_neg = sum(nonstop['neg'].values())
        neg_denom = sum_neg + smoothness*(len(set(nonstop['neg'])) + 1)
    for word in nonstop['pos'].copy():
        likelihood['pos'][word] = np.divide((nonstop['pos'][word] + smoothness), pos_denom)
        likelihood['pos']['OOV'] = np.divide(smoothness, pos_denom)
    for word in nonstop['neg'].copy():
        likelihood['neg'][word] = np.divide((nonstop['neg'][word] + smoothness), neg_denom)
        likelihood['neg']['OOV'] = np.divide(smoothness, neg_denom)
    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    # raise RuntimeError("You need to write this part!")
    # import math
    hypotheses = [0] * len(texts)
    # prob = {'pos':{}, 'neg':{}}
    p_pos = prior
    p_neg = 1-prior
    for i in range(len(texts)):
        sum_pos = 0
        sum_neg = 0
        for word in texts[i]:
            if word in stopwords:
                continue
            elif word not in likelihood['pos']:
                sum_pos += np.log(likelihood['pos']['OOV'])
            else:
                sum_pos += np.log(likelihood['pos'][word])
            if word not in likelihood['neg']:
                sum_neg += np.log(likelihood['neg']['OOV'])
            else:
                sum_neg += np.log(likelihood['neg'][word])
        if np.log(p_pos) + sum_pos > np.log(p_neg) + sum_neg:
                hypotheses[i] = 'pos'
        else:
            hypotheses[i] = 'neg'
    return hypotheses


def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    # raise RuntimeError("You need to write this part!")
    accuracies = np.zeros(shape=(len(priors), len(smoothnesses)))
    for i in range(len(smoothnesses)):
        for j in range(len(priors)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[i])
            hypotheses = naive_bayes(texts, likelihood, priors[j])
            count_correct = 0
            for (y,yhat) in zip(labels, hypotheses):
                if y==yhat:
                    count_correct += 1           
            accuracies[i][j] = count_correct / len(labels)
    return accuracies
    
                          
