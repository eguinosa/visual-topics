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


def model_homogeneity_per_size(
        model_id: str, size_list: list = None, show_progress=False
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
            parallelism=True, show_progress=show_progress
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
    # Create Dictionary with Homogeneity Values.
    print(f"Calculating the Homogeneity of the Model<{_model_id}>...")
    _homogeneity_dict = model_homogeneity_per_size(
        model_id=_model_id, show_progress=True
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
    _homogeneity_file = f"{_model_id}_homogeneity.json"
    _homogeneity_path = join('temp_data', _homogeneity_file)
    print("Saving the Homogeneity Values in the Temp Data Folder:")
    print(f"File -> {_homogeneity_file}")
    with open(_homogeneity_path, 'w') as f:
        json.dump(_homogeneity_dict, f)

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
