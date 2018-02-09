"""
Gensim: Corpora and vector spaces
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))
from gensim import corpora

# create is a tiny corpus of nine documents, each consisting of only a single sentence.
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
documents[1]
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

# To convert documents to vectors,  we’ll use a document representation
# called bag-of-words. each document is represented by one vector where
# each vector element represents a key-value pair of the number
# of times that word appears in the document

# represent the questions only by their (integer) ids.
# the mapping between the questions and ids is called a dictionary:
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict') ## tmp storage of dictionary
dictionary.token2id

# convert a new document to a vector representation using doc2bow function
# which simply counts the number of occurrences of each distinct word,
# converts the word to its integer word id and returns the result as a
# bag-of-words--a sparse vector, in the form of [(word_id, word_count), ...].

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored


# now that we know how to do convert one document to a sparse vector representation,
# let's do it to all of the documents
corpus = [dictionary.doc2bow(text) for text in texts]
