#!/usr/bin/env python
import sys
#Author: Serena Killion

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
import string


def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1: return a set of possible subsitutes 
    possible_synonyms = []
    s1 = wn.synsets(lemma, pos)
    for s in s1:
        for syn in s.lemma_names():
            print(syn)
            #output should not include the lemma itself
            if syn != lemma:
                if "_" in syn:
                    syn = syn.replace("_", " ")
                possible_synonyms.append(syn)

    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    #Part 2, predict highest possible synonyms for target word of context
     target = context.lemma
     #print(context.word_form)
     s1 = wn.synsets(target, context.pos)
     freq_dict = {}
     for s in s1:
         for syn in s.lemmas():
             word = syn.name().lower()
             if "_" in word:
                 word = word.replace("_", " ")
             if word != target:
                 if word not in freq_dict:
                     freq_dict[word] = syn.count() 
                 else:
                     freq_dict[word] += syn.count()  
     #get highest possible synonym (max value)
     val = list(freq_dict.values())
     key = list(freq_dict.keys())
     return key[val.index(max(val))]
 
        

def wn_simple_lesk_predictor(context):
    s1 = wn.synsets(context.lemma, context.pos)
    #print(f"{context.left_context} + *{context.word_form}* + {context.right_context}")
    contexts = set(context.left_context + context.right_context)
    stop_words = set(stopwords.words('english')) 
    count = 0
    save = None
    best_syn = None
    for s in s1:
        #get definitions, examples, lemmas, hypernyms
        d = tokenize(s.definition())
        compare = set(d)-stop_words
        compare = compare.union(set(s.lemma_names()))
        for ex in s.examples():
            ex = tokenize(ex)
            tmp = set(ex)-stop_words
            compare = compare.union(tmp)
        
        for hyp in s.hypernyms():
            h_def = tokenize(hyp.definition())
            h_def = set(h_def)-stop_words
            compare = compare.union(h_def)
            compare = compare.union(set(hyp.lemma_names()))
            
        #compute overlap with context words
        if len(compare.intersection(contexts)) > count:
            count = len(compare.intersection(contexts))
            save = s
        
        #return most freq synonym
        i = 0
        if save:
            for s in save.lemmas():
                #print(f"s.name: {s.name()}")
                #print(f"c.lemma: {context.lemma}")
                check1 = (s.name() == context.lemma)
                check2 = (s.name() == context.word_form)
                if i <= s.count() and s.name() and not check1 and not check2:
                    i = s.count()
                    best_syn = s.name()
        #if best_syn is None, return most freq. synonym
        if not best_syn:
            return wn_frequency_predictor(context)
        else:
            if "_" in best_syn:
                best_syn = best_syn.replace("_", " ")
            return best_syn
     

   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        #return syn most similar to target word
        possible_syns = get_candidates(context.lemma, context.pos)
        best_syn = None
        count = -1
        if possible_syns:
            for ps in possible_syns:
                if " " in ps:
                    ps.replace(" ", "_")
                if ps in self.model.vocab:
                    #similarity() returns value between 0 and 1
                    if count <= self.model.similarity(ps, context.lemma):
                        count = self.model.similarity(ps, context.lemma)
                        best_syn = ps
                try: 
                    if ps.capitalize() in self.model.vocab:
                        cap = ps.capitalize()
                        if count <= self.model.similarity(cap, context.lemma):
                            count = self.model.similarity(cap, context.lemma)
                            best_syn = ps
                except:
                    #print(f"{ps} not in vocab")
                    pass
        return best_syn
    
    
            
    def predict_nearest_with_context(self, context): 
        #to remove during preprocessing
        stop_words = set(stopwords.words('english')) 
        stop_punct =  set(string.punctuation)
        
        #preprocess left side
        left = context.left_context 
        left_side = []
        left_side = set(left_side)
        for w in left:
            left_side.add(w.lower())
        left_side = left_side - stop_words - stop_punct
        
        left_side = list(left_side)
        #use only last 5 words from right
        tmp = len(left_side) - 5
        if tmp > 0:
            for i in range(0, tmp):
                del left_side[0]
        
        #preprocess right side
        right = context.right_context
        right_side = []
        right_side = set(right_side)
        for w in right:
            right_side.add(w.lower())
        right_side = right_side - stop_words - stop_punct
        
        right_side = list(right_side)
        #use only last 5 words from left
        tmp2 = len(right_side) - 5
        if tmp2 > 0:
            for i in range(0, tmp2):
                right_side.pop()
        
        #combine both sides
        sentence = [*left_side, *right_side]
        full_context = []
        
        #check if words are in model.vocab
        for word in sentence:
            if word in self.model.vocab:
                full_context.append(word)
         
        #create singel vector with left, right context and target word
        v = np.zeros(len(self.model.wv[context.lemma])) + self.model.wv[context.lemma]
        for w in full_context:
            v += self.model.wv[w]
        
        #compute cosine similarity
        def cos(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            
        possible_syns = get_candidates(context.lemma, context.pos)
        count = -1
        best_syn = None
            
        #calculate similarity
        if possible_syns:
            for ps in possible_syns:
                if " " in ps:
                    ps = ps.replace(" ", "_")
                try: 
                    if ps in self.model.vocab:
                        if count <= cos(v, self.model.wv[ps]):
                            count = cos(v, self.model.wv[ps])
                            best_syn = ps
                except:
                    #print(f"{ps} not in vocab")
                    pass
        return best_syn



    def improved_predictor(self, context): 
        #to remove during preprocessing
        stop_words = set(stopwords.words('english')) 
        stop_punct =  set(string.punctuation)
        
        #preprocess left side
        left = context.left_context 
        left_side = []
        left_side = set(left_side)
        for w in left:
            left_side.add(w.lower())
        left_side = left_side - stop_words - stop_punct
        
        left_side = list(left_side)
        
        #use only last 5 words from right
        tmp = len(left_side) - 5
        if tmp > 0:
            for i in range(0, tmp):
                del left_side[0]
        
        #preprocess right side
        right = context.right_context
        right_side = []
        right_side = set(right_side)
        for w in right:
            right_side.add(w.lower())
        right_side = right_side - stop_words - stop_punct
        
        right_side = list(right_side)
        
        #use only last 5 words from left
        tmp2 = len(right_side) - 5
        if tmp2 > 0:
            for i in range(0, tmp2):
                right_side.pop()
        
        #combine both sides
        sentence = [*left_side, *right_side]
        full_context = []
        
        #check if words are in model.vocab
        for word in sentence:
            if word in self.model.vocab:
                full_context.append(word)
         
        #create singel vector with left, right context and target word
        v = np.zeros(len(self.model.wv[context.lemma])) + self.model.wv[context.lemma]
        for w in full_context:
            v += self.model.wv[w]
        
        #compute cosine similarity
        def cos(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        
            
        possible_syns = get_candidates(context.lemma, context.pos)
        count = -1
        best_syn = None
            
        #calculate similarity accounting for capitalized words 
        if possible_syns:
            for ps in possible_syns:
                if " " in ps:
                    ps = ps.replace(" ", "_")
                if ps in self.model.vocab:
                    if count <= cos(v, self.model.wv[ps]):
                        count = cos(v, self.model.wv[ps])
                        best_syn = ps
                try:
                    if ps.capitalize() in self.model.vocab:
                        cap = ps.capitalize()
                        if count <= cos(v, self.model.wv[cap].capitalize()):
                            count = cos(v, self.model.wv[cap].capitalize())
                            best_syn = ps
                except:
                    #print(f"{ps} not in vocab")
                    pass
        return best_syn
        

        
        
if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    

    
    for context in read_lexsub_xml(sys.argv[1]):
        
        #print(context)  # useful for debugging
        #prediction = wn_simple_lesk_predictor(context)
        #print(prediction)
        #prediction = wn_simple_lesk_predictor(context)
        prediction = predictor.improved_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))


    
