from sent2vec.vectorizer import Vectorizer
from scipy import spatial
import numpy as np
import pandas as pd
import time

sentence_data = pd.read_pickle("pickled_sentences")

sample_size = 70
min_difficulty = 3
pull = -0.7
push = 1
col_names = ["sub_type" , "difficulty_level" , "selected_sentence"]

nitem = sentence_data.index.size
rand_row = np.random.randint(0, nitem)

player_position = np.asarray(sentence_data.loc[rand_row, "sentence_vectors"])

sentence_choice_df = pd.DataFrame(columns = col_names)

#print(player_position)

keep_going = True
last_row = 0

for j in range(1,100):
    sentence_dist = 1
    selected_sentence = None

    for i in range(1,sample_size):

        rand_row = np.random.randint(0, nitem)

        if not rand_row == last_row:

            candidate_type = sentence_data.loc[rand_row, "sub_type"]
            candidate_sentence = sentence_data.loc[rand_row, "sentence_corpus"]
            candidate_difficulty = sentence_data.loc[rand_row, "difficulty_level"]
            candidate_vector = np.asarray(sentence_data.loc[rand_row, "sentence_vectors"])
            candidate_dist = spatial.distance.cosine(candidate_vector, player_position)

        if (candidate_dist < sentence_dist):

            last_row = rand_row

            selected_type = candidate_type
            selected_sentence = candidate_sentence
            selected_difficulty = candidate_difficulty
            selected_vector = candidate_vector
            sentence_dist = candidate_dist

    if selected_difficulty < min_difficulty:
       direction = push
    else:
        direction = pull
    # coin = np.random.randint(0,1)
    # if coin == 1:
    #     direction = push
    # elif coin == 0:
    #     direction = pull

    player_position = player_position + (direction * (player_position - selected_vector))

    new_row = {"sub_type": selected_type, "difficulty_level": selected_difficulty, "selected_sentence": selected_sentence}

    sentence_choice_df = sentence_choice_df.append(new_row, ignore_index=True)

    #print(selected_type, selected_difficulty, selected_sentence)

    #time.sleep(.1)

print(sentence_choice_df[-30:])