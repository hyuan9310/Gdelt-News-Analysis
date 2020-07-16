
import pandas as pd
import numpy as np
from newspaper import Article
import nltk

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity 
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
### for summarization
from gensim.summarization.summarizer import summarize as gensim_summarize 


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from langdetect import detect
import dill 

class ProcessPipeline:
	def __init__(self,texts=None,steps=["langdetection","summarization",'tokenization'],
		tokenization_steps=['remove_digits','remove_punctuation',"remove_stopwords",'lemmatization','stemmization'],stopwordsWhiteList=["n't", "not", "no"]):
		self.texts = texts
		self.steps = steps
		self.tokenization_steps = tokenization_steps
		self.stopwordsWhiteList = stopwordsWhiteList

	def process(self,text,return_str=False):
		if "langdetection" in self.steps:


			lang = self.detect_lang(text)
			if lang == "en":
				processed =  text
			else:
				processed = ""
		else:
			processed = text
		if "summarization"	in self.steps:

			processed = self.summarize(processed)
		if "tokenization" in self.steps:
			processed = self.pre_process(processed,return_str=return_str,steps=self.tokenization_steps,whiteList= self.stopwordsWhiteList)
		return processed
 

	def run(self,return_str=False,workers=6):
	    with ProcessPoolExecutor(max_workers=workers) as executor:
	    	if return_str:
	        	res = executor.map(self.process, self.texts,[True]*len(self.texts))        		
	    	else:
	        	res = executor.map(self.process, self.texts)
	    return list(res)    

		        
	def run_lambda(self):
	    return list(map(lambda x:self.process(x),self.texts))

	def run_loop(self):
	    processed = []
	    for i in self.texts:
	        processed.append(self.process(i))
	    return processed

	def detect_lang(self,text):
	    ### translate to english
	    try:
	        language = detect(text)
	        # print(f"language is {language}")
	    except:
	        # print("Not able to detect language")
	        language = "other"
	    return language

	def summarize(self,text,**kwargs):
	    try:
	        summarized = gensim_summarize(text,**kwargs)
	        return summarized
	    except:
	        return text

	def pre_process(self,text,return_str=False,steps=['remove_digits','remove_punctuation',"remove_stopwords",'lemmatization','stemmization'],whiteList=["n't", "not", "no"]):
	    ### Remove number: for func `translate`: yourstring.translate(str.maketrans(fromstr, tostr, deletestr))
	    if "remove_digits" in steps:
	    	text = text.translate(str.maketrans('', '',string.digits))
	    ### Remove punctuation
	    if "remove_punctuation"  in steps:
	    	text = text.translate(str.maketrans('', '', string.punctuation))

	    ### Remove stops words
	    if "remove_stopwords" in steps:

	    	text = [word for word in text.split() if (word.lower() not in stopwords.words('english') and not word.startswith("http")) or (word.lower() in whiteList)]

	    if "lemmatization" in steps:
	    	lemmatizer = nltk.WordNetLemmatizer()
	    	text = list(map(lambda x:lemmatizer.lemmatize(x),text))
	    ### Remove stemmization
	    if "stemmization" in steps:
	    	stemmer = PorterStemmer()
	    	text = list(map(lambda x:stemmer.stem(x),text))

	    if return_str:
	    	return (' ').join(text)
	    else:
	    	return text



def get_text(url):
    """
    Func: 1. get raw text from url 2. get summary & keyword from text
        Input: url, a link to article
        Output: dictionary contains 3 keys, text, summary & keywords
    """
    try:
        article = Article(url)
        article.download()

        ### parse html file
        article.parse()
        text = article.text
    
        return text
    except:
        print(f'fail to download news from {url}')
        return ""