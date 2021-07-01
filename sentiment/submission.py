#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    word_dict=collections.defaultdict(int)
    for word in x.split():
        word_dict[word]+=1
    return word_dict
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters=25, eta=0.1):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    
    def predictor(test):
        if dotProduct(weights,featureExtractor(test)) < 0:
            return -1
        return 1
    
    for x, y in trainExamples:
        for feature in featureExtractor(x):
           weights[feature]  = 0
           
    #print(weights)
    for t in range(numIters):
        for x,y in trainExamples:
           if (dotProduct(weights,featureExtractor(x))*y<1): #cost function is max(0, 1-dot(weights, phi))
               increment(weights, eta*y , featureExtractor(x))
               #print(weights)
               
    #print(evaluatePredictor(testExamples,predictor))
               
    # END_YOUR_CODE
    return weights

trainExamples = (("hello world", 1), ("goodnight moon", -1))
testExamples = (("hello", 1), ("moon", -1))
featureExtractor = extractWordFeatures
weights = learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        for feature in random.sample(list(weights), len(weights) - 1):
            phi = {feature:random.random()}
        if dotProduct(weights, phi) > 0:   # y should be 1 or -1 as classified by the weight vector. 
            y = 1
        else:
            y = -1
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        word_dict=collections.defaultdict(int)
        x=x.replace(" ","")
        for i in range(len(x)-n+1):
            word_dict[x[i:i+n]]+=1
        return word_dict
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
    return extract

#fe = extractCharacterFeatures(3)
#sentence = "hello world"
#ans = {"hel":1, "ell":1, "llo":1, "low":1, "owo":1, "wor":1, "orl":1, "rld":1}
#fe(sentence)

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    #print(examples)
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    def distance(x,mu):
        dis = 0
        for i in x:
            dis+=(x[i] - mu[i])**2
        return dis
        
    
    centers = random.sample(examples, K)
    z = [0] * len(examples) ## assigned cluster
    #print(z)
    
    for iters in range(maxIters):
        # step 1
        for i, x in enumerate(examples): #(0, {0: 0, 1: 0})
            #print(i) 
            min_d = float('inf')
            for k, mu in enumerate(centers): #(centroid)
                d = distance(x,mu)
                if d < min_d:
                    min_d = d
                    z[i] = k
        # step 2
        for k, mu in enumerate(centers):
            new_centroid = collections.defaultdict(float) #defaultdict(<class 'float'>, {0: 0.0, 1: 1.0})
            point_in_cluster = z.count(k)
            for i, x in enumerate(examples):
                if z[i] == k:
                    increment(new_centroid,1/point_in_cluster,x)
            centers[k] = new_centroid
                    
    #loss
    loss = 0
    for i, x in enumerate(examples):
        diff = x.copy() # {0: 0, 1: 0}
        #centers[z[i]]  #defaultdict(<class 'float'>, {0: 0.0, 1: 1.0})
        increment(diff,-1,centers[z[i]]) #diff - centroid
        #print(diff)  {0: 0.0, 1: -1.0}
        loss += dotProduct(diff,diff)

    return (centers, z, loss)
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE
