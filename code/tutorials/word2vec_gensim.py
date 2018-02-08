import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
           [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
           [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
           [(0, 1.0), (4, 2.0), (7, 1.0)],
           [(3, 1.0), (5, 1.0), (6, 1.0)],
           [(9, 1.0)],
           [(9, 1.0), (10, 1.0)],
           [(9, 1.0), (10, 1.0), (11, 1.0)],
           [(8, 1.0), (10, 1.0), (11, 1.0)]]


## transformation: convert documents from one vector representation into another
tfidf = models.TfidfModel(corpus)

vec = [(0, 1), (4, 1)]
print(tfidf[vec])


#  transform the whole corpus via TfIdf and index it,
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)

# query the similarity of our query vector vec against every document in the corpus:
# Document number zero (the first document) has a similarity score of 0.466=46.6%,
# the second document has a similarity score of 19.1% etc.
# Thus, according to TfIdf document representation and cosine similarity
# measure, the most similar to our query document vec is document no. 3,
# with a similarity score of 82.1%
sims = index[tfidf[vec]]
print(list(enumerate(sims)))
