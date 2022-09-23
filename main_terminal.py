# Gelin Eguinosa Rosique
# 2022

import sys
from mono_topics import MonoTopics
from time_keeper import TimeKeeper
from extra_funcs import big_number


if __name__ == '__main__':
    # Record Runtime of the Program.
    stopwatch = TimeKeeper()
    # Terminal Parameters.
    args = sys.argv

    # Load the Available Topic Models.
    model_ids = MonoTopics.saved_models()
    print("\nAvailable Topic Models:")
    for model_id in model_ids:
        print(f"\nTopic Model <{model_id}>:")
        basic_info = MonoTopics.basic_info(model_id)
        # Display Basic Info of the Model.
        print("  Topics:", big_number(basic_info['topic_size']))
        print("  Corpus Size:", big_number(basic_info['corpus_size']))
        print("  Document Model:", basic_info['text_model_name'].replace('_', ' ').title())
        hierarchy = 'Yes' if basic_info['has_reduced_topics'] else 'No'
        print("  Topic Hierarchy:", hierarchy)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
