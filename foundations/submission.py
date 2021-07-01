import collections
import math

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return max(text.split())
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt((loc2[0]-loc1[0])**2+(loc2[1]-loc1[1])**2)
    # END_YOUR_CODE

############################################################
# Problem 3c -- failure

def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    similar = []
    next_words=collections.defaultdict(set)
    words = sentence.split()
    for i in range(len(words)-1):
        next_words[words[i]].add(words[i+1])
    #print(next_words)
    
    def recurse(sent):
        if len(sent) == len(words):
            similar.append(' '.join(sent))
        else:
            for next_word in next_words[sent[-1]]:
                recurse(sent+[next_word]) 
        
    for word in set(next_words):
        recurse([word])

    #print(similar)

    return similar      
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE
#mutateSentences('the cat')

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    DotProduct = 0
    for v in v1:
        DotProduct += v1[v]*v2[v]
               
    return DotProduct
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE
    
############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for u in v2:
        v1[u] += scale*v2[u]

    return v1
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    ans=set()
    words=collections.Counter(text.split())
    for word in words:
        if words[word] ==1:
            ans.add(word)
        

    return ans
    
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE
findSingletonWords('the quick brown fox jumps over the lazy fox')
############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    reverse = text[::-1]
    
    T=[]
    for a in range(len(text)+1):
        row = [0]
        for b in range(1,len(reverse)+1):
            if a==0:
                row.append(0)
            else:
                row.append(0)
        T.append(row)
    
    #print(T)

    for i in range(1,len(text)+1):
        for j in range(1,len(reverse)+1):
            if text[i-1] == reverse[j-1]:
                T[i][j] = T[i-1][j-1]+1
            else:
                T[i][j] = max(T[i-1][j],T[i][j-1])
                


    return T[-1][-1]
    
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

computeLongestPalindromeLength('text')
