# Gelin Eguinosa Rosique
# 2022

import json
from os.path import join
# import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt

from extra_funcs import number_to_digits


# Topic Model's Files.
temp_folder = 'temp_data'
homogeneity_files = {
    'mono_homogeneity': 'sbert_fast_105_548_docs_745_topics_homogeneity.json',
    'mix_homogeneity': 'specter_sbert_fast_105_548_docs_533_topics_homogeneity.json',
}
pwi_files = {
    'mono_exact_20': 'sbert_fast_105_548_docs_745_topics_pwi_exact_20_words.json',
    'mono_exact_50': 'sbert_fast_105_548_docs_745_topics_pwi_exact_50_words.json',
    'mono_tf-idf_20': 'sbert_fast_105_548_docs_745_topics_pwi_tf-idf_20_words.json',
    'mono_tf-idf_50': 'sbert_fast_105_548_docs_745_topics_pwi_tf-idf_50_words.json',
    'mix_exact_20': 'specter_sbert_fast_105_548_docs_533_topics_pwi_exact_20_words.json',
    'mix_exact_50': 'specter_sbert_fast_105_548_docs_533_topics_pwi_exact_50_words.json',
    'mix_tf-idf_20': 'specter_sbert_fast_105_548_docs_533_topics_pwi_tf-idf_20_words.json',
    'mix_tf-idf_50': 'specter_sbert_fast_105_548_docs_533_topics_pwi_tf-idf_50_words.json',
}


def pwi_plot_data(data_type='exact_20'):
    """
    Create the data to plot the comparison between the Mono and Mix Topic
    Models. There 4 types of comparison:
     - 'exact_20' containing the models PWI-exact using the top 20 words.
     - 'exact_50' containing the models PWI-exact using the top 50 words.
     - 'tf-idf_20' containing the models PWI-tf-idf using the top 20 words.
     - 'tf-idf_50' containing the models PWI-tf-idf using the top 50 words.

    Returns: X_Values, Mono_Values, Mix_Values - three lists with the topic
        sizes, the PWI values of the Mono Topics and the PWI values of the Mix
        Topics.
    """
    # Create X Values.
    x_values = [10 * x for x in range(1, 11)]

    # Load Mono Dictionary.
    mono_dict_file = pwi_files[f"mono_{data_type}"]
    mono_dict_path = join(temp_folder, mono_dict_file)
    with open(mono_dict_path, 'r') as f:
        mono_dict = json.load(f)
    # Create Mono Values.
    mono_values = [
        round(1000 * mono_dict[number_to_digits(size, 3)], 1)
        for size in x_values
    ]

    # Load Mix Dictionary.
    mix_dict_file = pwi_files[f"mix_{data_type}"]
    mix_dict_path = join(temp_folder, mix_dict_file)
    with open(mix_dict_path, 'r') as f:
        mix_dict = json.load(f)
    # Create Mix Values.
    mix_values = [
        round(1000 * mix_dict[number_to_digits(size, 3)], 1)
        for size in x_values
    ]

    # Add Values for Zero.
    x_values = [0] + x_values
    mono_values = [0.0] + mono_values
    mix_values = [0.0] + mix_values

    # Plotting Data.
    return x_values, mono_values, mix_values


def homogeneity_plot_data():
    """
    Create the data to plot the Homogeneity comparison between the Mono and Mix
    Topic Models.

    Returns: X_values, Mono_values, Mix_values - three lists with the topic
        sizes, the Homogeneity values of the Mono Topics and the Homogeneity
        values of the Mix Topics.
    """
    # Create X Values.
    x_values = [10 * x for x in range(1, 11)]

    # Load Mono Data.
    mono_dict_file = homogeneity_files['mono_homogeneity']
    mono_dict_path = join(temp_folder, mono_dict_file)
    with open(mono_dict_path, 'r') as f:
        mono_dict = json.load(f)
    # Create Mono Values.
    mono_values = [
        round(mono_dict[number_to_digits(size, 3)], 2)
        for size in x_values
    ]

    # Load Mix Data.
    mix_dict_file = homogeneity_files['mix_homogeneity']
    mix_dict_path = join(temp_folder, mix_dict_file)
    with open(mix_dict_path, 'r') as f:
        mix_dict = json.load(f)
    # Create Mix Values.
    mix_values = [
        round(mix_dict[number_to_digits(size, 3)], 2)
        for size in x_values
    ]

    # Add Values for Zero.
    x_values = [0] + x_values
    mono_values = [0.0] + mono_values
    mix_values = [0.0] + mix_values

    # Plotting Data.
    return x_values, mono_values, mix_values


if __name__ == '__main__':
    # # --- Create PWI Plot ---
    # _x_values, _mono_values, _mix_values = pwi_plot_data('tf-idf_50')
    # # _x_values, _mono_values, _mix_values = pwi_plot_data('exact_50')
    # # --------------------------------------------------------------
    # # Doc2Vec Values.
    # _doc2vec_tfidf = [0.0, 10.1, 14.4, 18.8, 21.8, 24.5, 25.2, 26.1, 27.6, 28.9, 30.7]
    # _doc2vec_exact = [0.0, 11.5, 16.4, 21.4, 24.8, 27.9, 28.8, 29.7, 31.5, 32.9, 34.9]
    # _doc2vec_values = _doc2vec_tfidf
    # # _doc2vec_values = _doc2vec_exact
    # # --------------------------------------------------------------
    # # LDA Values
    # _lda_tfidf = [0.0, 2.5, 3.2, 3.6, 4.5, 5.3, 6.5, 7.1, 7.2, 7.8, 9.0]
    # _lda_exact = [0.0, 2.9, 3.7, 4.1, 5.1, 6.0, 7.4, 8.0, 8.2, 8.9, 10.3]
    # _lda_values = _lda_tfidf
    # # _lda_values = _lda_exact
    # # --------------------------------------------------------------
    # fig, ax = plt.subplots()
    # ax.plot(_x_values, _mono_values, label='Sentence-BERT')
    # ax.plot(_x_values, _mix_values, label='SPECTER-SBERT')
    # ax.plot(_x_values, _doc2vec_values, label='Doc2Vec')
    # ax.plot(_x_values, _lda_values, label='LDA')
    # ax.plot()
    # ax.set_xlabel("Número de Tópicos")
    # ax.set_ylabel("Ganancia de Información")
    # ax.set_title("Comparación con PWI-tf-idf (50 palabras)")
    # # ax.set_title("Comparación con PWI-exact (50 palabras)")
    # # ax.axis([0, 100, 0, max(max(_mono_values), max(_mix_values))])
    # ax.axis([0, 100, 0, 110])
    # ax.legend()
    # # --------------------------------------------------------------
    # # plt.show()
    # plt.savefig('temp_data/comparison_pwi_tf-idf_50_words.pdf', format='pdf')
    # # plt.savefig('temp_data/comparison_pwi_exact_50_words.pdf', format='pdf')

    # --- Create Homogeneity Plot ---
    _x_values, _mono_values, _mix_values = homogeneity_plot_data()
    # --------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(_x_values, _mono_values, label='Sentence-BERT')
    ax.plot(_x_values, _mix_values, label='SPECTER-SBERT')
    ax.plot()
    ax.set_xlabel("Número de Tópicos")
    ax.set_ylabel("Homogeneidad")
    ax.set_title("Comparación de los modelos Homogeneidad")
    ax.axis([0, 100, 0, max(max(_mono_values), max(_mix_values))])
    ax.legend()
    # --------------------------------------------------------------
    plt.savefig("temp_data/comparison_homogeneity.pdf", format='pdf')
