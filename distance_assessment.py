from sent2vec.vectorizer import Vectorizer
from scipy import spatial
import numpy as np
import pandas as pd
import time

sentence_data = pd.read_pickle("pickled_sentences")

sample_size = 5
min_difficulty = 3
pull = -0.7
push = 1

player_position = np.asarray(sentence_data.loc[0, "sentence_vectors"])

#print(player_position)

keep_going = True

while keep_going:
    sentence_dist = 1
    selected_sentence = None

    nitem = sentence_data.index.size

    for i in range(1,sample_size):

        rand_row = np.random.randint(0, nitem)

        candidate_sentence = sentence_data.loc[rand_row, "sentence_corpus"]
        candidate_difficulty = sentence_data.loc[rand_row, "difficulty_level"]
        candidate_vector = np.asarray(sentence_data.loc[rand_row, "sentence_vectors"])
        candidate_dist = spatial.distance.cosine(candidate_vector, player_position)

        if (candidate_dist < sentence_dist):

            selected_sentence = candidate_sentence
            selected_difficulty = candidate_difficulty
            selected_vector = candidate_vector
            sentence_dist = candidate_dist

    if selected_difficulty < min_difficulty:
       direction = pull
    else:
        direction = push
        # coin = np.random.randint(0,1)
        # if coin == 1:
        #     direction = push
        # elif coin == 0:
        #     direction = pull

    player_position = player_position + (direction * (player_position - selected_vector))

    print(selected_difficulty, selected_sentence)

    time.sleep(.1)