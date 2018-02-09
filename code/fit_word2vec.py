"""
Gensim: Fit Word2Vec Model
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))
from gensim import corpora
from collections import defaultdict

# read raw txt data
with open('data/childes_test.txt') as f:
    documents = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
documents = [x.strip() for x in content]
documents = [x.replace('"', '') for x in content]

# process the raw text data
stoplist = set('for a of the and to in'.split())
texts = []
for document in documents:
    text = document.split()
    text = [word.lower() for word in text if word not in stoplist]
    texts.append(text)

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

# represent the questions only by their (integer) ids.
# the mapping between the questions and ids is called a dictionary:
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/childes.dict')
dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in texts]

# check out the corpus object
corpus[0:10]
