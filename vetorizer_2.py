from sent2vec.vectorizer import Vectorizer
from scipy import spatial
import numpy as np
import pandas as pd
import time
import pickle

sentence_data = pd.read_csv("./data/tasks/sentence_correction/task_data.csv")

print(sentence_data.columns)

start_time = time.time()

whole_sentences = []

for row, values in sentence_data.iterrows():
    whole_sentences.append(values[2].format(values[3].strip("{}")))

sentence_data["sentence_corpus"] = whole_sentences

vectorizer = Vectorizer()
vectorizer.bert(sentence_data["sentence_corpus"])
sentence_data["sentence_vectors"] = (vectorizer.vectors).tolist()

print(sentence_data.index)

end_time = time.time() - start_time

print(end_time)

sentence_data.to_pickle("pickled_sentences")

# dist_1 = spatial.distance.cosine(vectors[3], vectors[0])
# dist_2 = spatial.distance.cosine(vectors[3], vectors[2])
# print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))