'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    word0_max = 0
    word1_max = 0
    for line in texts:
      X0 = 0
      X1 = 0
      for word in line:
        if word == word0:
          X0+=1
        if word == word1:
          X1+=1
      word0_max = max(word0_max, X0)
      word1_max = max(word1_max, X1)

    word0_max+=1
    word1_max+=1
    x = np.arange(word0_max*word1_max)
    x = x.reshape((word0_max, word1_max))
    Pjoint = np.zeros_like(x)
    for line in texts:
      X0 = 0
      X1 = 0
      for word in line:
        if word == word0:
          X0+=1
        if word == word1:
          X1+=1
      Pjoint[X0][X1]+=1
      # return np.divide(Pjoint, len(texts))
    Pjoint = Pjoint/len(texts)
    # raise RuntimeError('You need to write this part!')
    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    Pmarginal = []
    if index == 0:
      for row in range(len(Pjoint)):
        Pmarginal.append(sum(Pjoint[row]))

    if index == 1:
      # for i in range(0, len(Pjoint)):
      #   sumcol = 0
      #   for j in range(0, len(Pjoint[0])-1):
      #     sumcol += Pjoint[j][i]
          
      #   Pmarginal.append(sumcol)
      Pmarginal = np.sum(Pjoint, axis=0)
      
    # raise RuntimeError('You need to write this part!')
    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    np.seterr(invalid='ignore')
    Pcond = np.divide(Pjoint.T, Pmarginal).T
    # raise RuntimeError('You need to write this part!')
    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    # raise RuntimeError('You need to write this part!')
    mu = 0
    for i in range(len(P)):
      mu += i*P[i]
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    # raise RuntimeError('You need to write this part!')
    var = 0
    EX = mean_from_distribution(P)
    for i in range(len(P)):
      var += (i-EX)**2 * P[i]
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    # raise RuntimeError('You need to write this part!')
    covar = 0
    EX = mean_from_distribution(marginal_distribution_of_word_counts(P, 0))
    EY = mean_from_distribution(marginal_distribution_of_word_counts(P, 1))
    for x in range(len(P)):
      for y in range(len(P[0])):
        covar += (x - EX) * (y - EY) * P[x][y]
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    # raise RuntimeError('You need to write this part!')
    expected = 0
    for X0 in range(len(P)):
      for X1 in range(len(P[0])):
        expected += f(X0, X1) * P[X0][X1]
    return expected
    
