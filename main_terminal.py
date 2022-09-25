# Gelin Eguinosa Rosique
# 2022

import sys
from mono_topics import MonoTopics
from mix_topics import MixTopics
from time_keeper import TimeKeeper
from extra_funcs import big_number


if __name__ == '__main__':
    # Record Runtime of the Program.
    stopwatch = TimeKeeper()
    # Terminal Parameters.
    args = sys.argv

    # Display the Available Mono Topic Models.
    model_ids = MonoTopics.saved_models()
    print("\n<<- Available Mono Topic Models ->>")
    for model_id in model_ids:
        print(f"\nMono Topic Model <{model_id}>:")
        basic_info = MonoTopics.basic_info(model_id)
        # Show the Basic Info of the Model.
        print("  Topics:", big_number(basic_info['topic_size']))
        print("  Corpus Size:", big_number(basic_info['corpus_size']))
        print("  Document Model:", basic_info['text_model_name'].replace('_', ' ').title())
        hierarchy = 'Yes' if basic_info['has_reduced_topics'] else 'No'
        print("  Topic Hierarchy:", hierarchy)

    # Display the Available Mix Topic Models.
    model_ids = MixTopics.saved_models()
    print("\n<<- Available Mix Topic Models ->>")
    for model_id in model_ids:
        print(f"\nMix Topic Model <{model_id}>:")
        basic_info = MixTopics.basic_info(model_id)
        # Show the basic Info of the Model.
        print("  Topics:", big_number(basic_info['topic_size']))
        print("  Corpus Size:", big_number(basic_info['corpus_size']))
        print("  Document Model: Specter")
        print("  Vocabulary Model:", basic_info['vocab_model_name'].replace('_', ' ').title())
        hierarchy = 'Yes' if basic_info['has_reduced_topics'] else 'No'
        print("  Topic Hierarchy:", hierarchy)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
