'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np
def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''

    distances = np.linalg.norm(train_images - image, axis=-1)
    sorted = np.argsort(distances)
    neighbors = train_images[sorted[:k]]
    labels = [train_labels[i] for i in sorted[:k]]
    return neighbors, labels

    # raise RuntimeError('You need to write this part!')


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    
    # hypotheses = []
    # scores = []
    # for image in dev_images:
    #     _, labels = k_nearest_neighbors(image, train_images, train_labels, k)
    #     unique, counts = np.unique(labels, return_counts=True)
    #     max_label = unique[np.argmax(counts)]
    #     hypotheses.append(max_label)
    #     scores.append(counts[np.argmax(counts)])
    # return hypotheses, scores

    hypotheses = []
    scores = []
    for image in dev_images:
        neighbors, labels = k_nearest_neighbors(image, train_images, train_labels, k)
        max_label = max(set(labels), key=labels.count)
        score = labels.count(max_label)
        if max_label == True:
            hypotheses.append(1)
        else:
            hypotheses.append(0)
        scores.append(score)
    return hypotheses, scores
    
    # raise RuntimeError('You need to write this part!')


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    confusions = np.zeros((2, 2))
    for i in range(len(references)):
        if hypotheses[i] == references[i] == True:
            confusions[1, 1]+=1
        elif hypotheses[i] == references[i] == False:
            confusions[0, 0]+=1
        elif hypotheses[i] == False and references[i] == True:
            confusions[1, 0]+=1
        else:
            confusions[0, 1]+=1
    tp = confusions[1, 1]
    fp = confusions[0, 1]
    fn = confusions[1, 0]
    tn = confusions[0, 0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2/ (1 / precision + 1 / recall)
    return confusions, accuracy, f1

    # raise RuntimeError('You need to write this part!')
