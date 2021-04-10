# Input: sentence data from "pickled_sentences"
# Output: dataframe of repeatedly selected sentences data
# With the setup parameters this does repeat a process of checking a sample of random sentence vectors for the nearest
# to the current player position. Then it either pushes or pulls the player vector nearer to selected vector and repeats
# the process. This is supposed to send the player in a cluster of sentences that are within or above his difficulty
# level.

from scipy import spatial
import numpy as np
import pandas as pd

# setup
sample_size = 70
repetitions = 100
min_difficulty = 3
pull = -0.7
push = 1
distance_threshold_max = 1

# init variables
sentence_data = pd.read_pickle("pickled_sentences")
col_names = ["sub_type", "difficulty_level", "selected_sentence"]
sentence_choice_df = pd.DataFrame(columns=col_names)

# init random player starting position
nitem = sentence_data.index.size
rand_row = np.random.randint(0, nitem)
player_position = np.asarray(sentence_data.loc[rand_row, "sentence_vectors"])

if __debug__:
    print(player_position)
# repeat the selection process and update the player position for set number of repetitions
# with a number of sample_size randomly selected sentences.
previous_row = 0
for j in range(1, repetitions):
    # reset variables
    sentence_dist = distance_threshold_max
    selected_sentence = None

    for i in range(1, sample_size):
        # choose random sentence
        rand_row = np.random.randint(0, nitem)

        if not rand_row == previous_row:
            # extract information of randomly selected sentences
            candidate_type = sentence_data.loc[rand_row, "sub_type"]
            candidate_sentence = sentence_data.loc[rand_row, "sentence_corpus"]
            candidate_difficulty = sentence_data.loc[rand_row, "difficulty_level"]
            candidate_vector = np.asarray(sentence_data.loc[rand_row, "sentence_vectors"])
            candidate_dist = spatial.distance.cosine(candidate_vector, player_position)

        # decide if sentence candidate is the nearest candidate, then select it
        if candidate_dist < sentence_dist:
            selected_type = candidate_type
            selected_sentence = candidate_sentence
            selected_difficulty = candidate_difficulty
            selected_vector = candidate_vector
            sentence_dist = candidate_dist

            previous_row = rand_row

    # pull or push player position from or to the new selected sentence position
    if selected_difficulty < min_difficulty:
        direction = push
    else:
        direction = pull
    player_position = player_position + (direction * (player_position - selected_vector))

    # save selected sentence to the dataframe
    new_row = {
        "sub_type": selected_type,
        "difficulty_level": selected_difficulty,
        "selected_sentence": selected_sentence
    }
    sentence_choice_df = sentence_choice_df.append(new_row, ignore_index=True)

    if __debug__:
        print(selected_type, selected_difficulty, selected_sentence)
        time.sleep(.1)
        print(sentence_choice_df[-30:])
