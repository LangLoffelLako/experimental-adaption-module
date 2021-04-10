# Creates a pickled file to read vectorized sentence data for "distance_assessment.py"
# Input: "./data/tasks/sentence_correction/task_data.csv"

from sent2vec.vectorizer import Vectorizer
import pandas as pd
import time

# setup
sentence_data = pd.read_csv("./data/tasks/sentence_correction/task_data.csv")
whole_sentences = []

if __debug__:
    print(sentence_data.columns)
    start_time = time.time()

# each "row" contains its "values" as list item
# save corrected sentences to "whole_sentences"
for row, values in sentence_data.iterrows():
    whole_sentences.append(values[2].format(values[3].strip("{}")))

sentence_data["sentence_corpus"] = whole_sentences

# create vectorized items and save them as list
vectorizer = Vectorizer()
vectorizer.bert(sentence_data["sentence_corpus"])
sentence_data["sentence_vectors"] = vectorizer.vectors.tolist()

if __debug__:
    print(sentence_data.index)
    end_time = time.time() - start_time
    print(end_time)

sentence_data.to_pickle("pickled_sentences")
