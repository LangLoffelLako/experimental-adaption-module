from sent2vec.vectorizer import Vectorizer
from scipy import spatial
import numpy as np
import time

sentences_1 = [
    "I like english.",
    "You are crazy",
    "I admire english to the greatest."
]

start_time = time.time()

vectorizer = Vectorizer()
vectorizer.bert(sentences_1)
vectors = vectorizer.vectors

vectors = np.concatenate([vectors, np.array([(vectors[2] + (vectors[2] - vectors[1]))])])

print(vectors)

end_time = time.time() - start_time

print(end_time)

dist_1 = spatial.distance.cosine(vectors[3], vectors[0])
dist_2 = spatial.distance.cosine(vectors[3], vectors[2])
print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))