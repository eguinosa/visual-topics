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


def create_plot_data(data_type='exact_20'):
    """
    Create the data to plot the comparison between the Mono and Mix Topics
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
    x_values = [2] + [10 * x for x in range(1, 11)]

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

    # Plotting Data.
    return x_values, mono_values, mix_values


if __name__ == '__main__':
    _x_values, _mono_values, _mix_values = create_plot_data('tf-idf_50')

    fig, ax = plt.subplots()
    ax.plot(_x_values, _mono_values, label='modelo sbert')
    ax.plot(_x_values, _mix_values, label='modelo specter-sbert')
    ax.set_xlabel('Número de Tópicos')
    ax.set_ylabel('Ganancia de Información')
    ax.set_title("Comparación con PWI-tf-idf (50 palabras)")
    ax.axis([0, 100, 0, max(max(_mono_values), max(_mix_values))])
    ax.legend()

    # plt.show()
    plt.savefig('temp_data/comparison_pwi_tf-idf_50_words.pdf', format='pdf')
    # plt.savefig('temp_data/power_plot_tight.pdf', format='pdf', bbox_inches='tight')
