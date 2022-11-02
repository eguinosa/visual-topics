# Gelin Eguinosa Rosique
# 2022

from mono_topics import MonoTopics
from mix_topics import MixTopics
from extra_funcs import progress_msg, number_to_digits

# Testing Imports.
import sys
import json
from os.path import join
from time_keeper import TimeKeeper


# Topic Model's Files.
temp_folder = 'temp_data'
homogeneity_files = {
    'mono_homog_doc-doc': 'sbert_fast_105_548_docs_745_topics_homogeneity_doc-doc.json',
    'mono_homog_topic-doc': 'sbert_fast_105_548_docs_745_topics_homogeneity_topic-doc.json',
    'mix_homog_doc-doc': 'specter_sbert_fast_105_548_docs_533_topics_homogeneity_doc-doc.json',
    'mix_homog_topic-doc': 'specter_sbert_fast_105_548_docs_533_topics_homogeneity_topic-doc.json',
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


def pwi_plot_data(data_type='exact_20', add_zero=True):
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

    # Check if we have to add values for Zero.
    if add_zero:
        x_values = [0] + x_values
        mono_values = [0.0] + mono_values
        mix_values = [0.0] + mix_values

    # Plotting Data.
    return x_values, mono_values, mix_values


def homogeneity_plot_data(data_type='homog_doc-doc', add_zero=True):
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
    mono_dict_file = homogeneity_files[f'mono_{data_type}']
    mono_dict_path = join(temp_folder, mono_dict_file)
    with open(mono_dict_path, 'r') as f:
        mono_dict = json.load(f)
    # Create Mono Values.
    mono_values = [
        round(mono_dict[number_to_digits(size, 3)], 2)
        for size in x_values
    ]

    # Load Mix Data.
    mix_dict_file = homogeneity_files[f'mix_{data_type}']
    mix_dict_path = join(temp_folder, mix_dict_file)
    with open(mix_dict_path, 'r') as f:
        mix_dict = json.load(f)
    # Create Mix Values.
    mix_values = [
        round(mix_dict[number_to_digits(size, 3)], 2)
        for size in x_values
    ]

    # Check if we have to add values for Zero.
    if add_zero:
        x_values = [0] + x_values
        mono_values = [0.0] + mono_values
        mix_values = [0.0] + mix_values

    # Plotting Data.
    return x_values, mono_values, mix_values


def model_homogeneity_per_size(
        model_id: str, homog_type: str, size_list: list = None, show_progress=False
):
    """
    Create a dictionary with the Homogeneity values of the Model for each of the
    sizes given in the 'size_list', if 'size_list' is None use by default the
    sizes 10, 20, ..., 90, 100.
    """
    # Load the Topic Model.
    if MonoTopics.model_saved(model_id=model_id):
        if show_progress:
            progress_msg("Loading a MonoTopics Model...")
        topic_model = MonoTopics.load(model_id=model_id, show_progress=show_progress)
    elif MixTopics.model_saved(model_id=model_id):
        if show_progress:
            progress_msg("Loading a MixTopics Model...")
        topic_model = MixTopics.load(model_id=model_id, show_progress=show_progress)
    else:
        raise NameError(f"The Topic Model <{model_id}> does not exist.")

    # Check we have a size list.
    if not size_list:
        model_size = topic_model.topic_size
        size_list = [
            topic_size for topic_size in topic_model.main_sizes(model_size)
            if topic_size <= 100 and topic_size % 10 == 0
        ]
        size_list.reverse()

    # Save in a dictionary the Homogeneity value for each size.
    homogeneity_dict = {}
    for size in size_list:
        # Reduce Topic Model.
        if show_progress:
            progress_msg(f"Updating the Size of the Model to {size}...")
        topic_model.reduce_topics(
            new_size=size, parallelism=True, show_progress=show_progress
        )
        # Get the Homogeneity Value for this size.
        homogeneity_value = topic_model.cur_model_homogeneity(
            homog_type=homog_type, parallelism=True, show_progress=show_progress
        )
        # Save the Homogeneity value.
        size_key = number_to_digits(size, 3)
        homogeneity_dict[size_key] = homogeneity_value
        if show_progress:
            progress_msg(
                f"Homogeneity Value for Topic Size <{size}>: {homogeneity_value}"
            )

    # Dictionary with the Homogeneity for the given sizes.
    return homogeneity_dict


def model_pwi_per_size(
        model_id: str, num_words=50, pwi_type='tf-idf', size_list: list = None,
        show_progress=False
):
    """
    Create a dictionary with the PWI values of the Model for each of the sizes
    given in the 'size_list', if 'size_list' is None use by default the sizes
    10, 20, ..., 90, 100.
    """
    # Load the Topic Model.
    if MonoTopics.model_saved(model_id=model_id):
        if show_progress:
            progress_msg("Loading a MonoTopics Model...")
        topic_model = MonoTopics.load(model_id=model_id, show_progress=show_progress)
    elif MixTopics.model_saved(model_id=model_id):
        if show_progress:
            progress_msg("Loading a MixTopics Model...")
        topic_model = MixTopics.load(model_id=model_id, show_progress=show_progress)
    else:
        raise NameError(f"The Topic Model <{model_id}> does not exist.")

    # Check we have a size list.
    if not size_list:
        model_size = topic_model.topic_size
        size_list = [
            topic_size for topic_size in topic_model.main_sizes(model_size)
            if topic_size <= 100 and topic_size % 10 == 0
        ]
        size_list.reverse()

    # Save in a dictionary the PWI value per each size.
    pwi_dict = {}
    for size in size_list:
        # Reduce Topic Model.
        if show_progress:
            progress_msg(f"Updating the Size of the Model to {size}...")
        topic_model.reduce_topics(
            new_size=size, parallelism=True, show_progress=show_progress
        )
        # Get the PWI value of the model.
        pwi_value = topic_model.cur_model_pwi(word_num=num_words, pwi_type=pwi_type)
        # Save PWI value.
        size_key = number_to_digits(size, 3)
        pwi_dict[size_key] = pwi_value
        if show_progress:
            progress_msg(f"PWI value with topic size <{size}>: {pwi_value}")

    # Dictionary with the PWI values.
    return pwi_dict


if __name__ == '__main__':
    # Record the Runtime of the Program.
    _stopwatch = TimeKeeper()
    # Terminal Parameters.
    _args = sys.argv

    # Select Model.
    # _model_id = 'sbert_fast_105_548_docs_745_topics'
    _model_id = 'specter_sbert_fast_105_548_docs_533_topics'
    # _model_id = 'sbert_fast_5_000_docs_55_topics'
    # _model_id = 'specter_sbert_fast_5_000_docs_25_topics'

    # --- Get the Homogeneity Values of the Topic Model ---
    # Homogeneity Type.
    # _homog_type = 'doc-doc'
    _homog_type = 'topic-doc'
    # --------------------------------------------------------------
    # Create Dictionary with Homogeneity Values.
    print(f"Calculating the Homogeneity of the Model<{_model_id}>...")
    _homogeneity_dict = model_homogeneity_per_size(
        model_id=_model_id, homog_type=_homog_type, show_progress=True
    )
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")
    # --------------------------------------------------------------
    # Show the Homogeneity Values.
    print("\nHomogeneity Values of the Topic Model:")
    _homogeneity_keys = list(_homogeneity_dict.keys())
    _homogeneity_keys.sort()
    for _size in _homogeneity_keys:
        if int(_size) < 10:
            print(f"   {_size[2:]}: {_homogeneity_dict[_size]}")
        elif int(_size) < 100:
            print(f"  {_size[1:]}: {_homogeneity_dict[_size]}")
        else:
            print(f" {_size}: {_homogeneity_dict[_size]}")
    # --------------------------------------------------------------
    # Save the Homogeneity Values of the Model.
    _homogeneity_file = f"{_model_id}_homogeneity_{_homog_type}.json"
    _homogeneity_path = join('temp_data', _homogeneity_file)
    print("Saving the Homogeneity Values in the Temp Data Folder:")
    print(f"File -> {_homogeneity_file}")
    with open(_homogeneity_path, 'w') as file:
        json.dump(_homogeneity_dict, file)

    # # --- Get the PWI Values of the Topic Model ---
    # # Select Number of Words.
    # _num_words = 20
    # # --------------------------------------------------------------
    # # Select PWI type.
    # _pwi_type = 'tf-idf'
    # # _pwi_type = 'exact'
    # # ------------------------------------------------------
    # # Create Dict with PWI values.
    # print(f"\nCalculating the PWI-{_pwi_type} of <{_model_id}>...")
    # _pwi_dict = model_pwi_per_size(
    #     model_id=_model_id, pwi_type=_pwi_type,
    #     num_words=_num_words, show_progress=True
    # )
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")
    # # ------------------------------------------------------
    # # Show the PWI Values.
    # print("\nPWI Values of the Topic Model:")
    # _dict_keys = list(_pwi_dict.keys())
    # _dict_keys.sort()
    # for _size in _dict_keys:
    #     if int(_size) < 10:
    #         print(f"   {_size[2:]}: {_pwi_dict[_size]}")
    #     elif int(_size) < 100:
    #         print(f"  {_size[1:]}: {_pwi_dict[_size]}")
    #     else:
    #         print(f" {_size}: {_pwi_dict[_size]}")
    # # ------------------------------------------------------
    # # Save the PWI Values of the Model.
    # _pwi_file = f"{_model_id}_pwi_{_pwi_type}_{_num_words}_words.json"
    # _pwi_path = join('temp_data', _pwi_file)
    # print(f"\nSaving the PWI Values in the Temp Data Folder:\nFile -> {_pwi_file}")
    # with open(_pwi_path, 'w') as f:
    #     json.dump(_pwi_dict, f)

    # Program Finished.
    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
