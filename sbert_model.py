# Gelin Eguinosa
# 2022

import sys
from os import mkdir
from os.path import isdir, join
from sentence_transformers import SentenceTransformer

from extra_funcs import progress_msg, big_number
from time_keeper import TimeKeeper

# Test Imports
from util_funcs import cos_sim


class SBertModel:
    """
    Load and Manage the Sentence-Bert pre-trained models to get the vector
    representations of the words and documents in the Cord-19 corpus.

    The models are available on:
    https://www.sbert.net/docs/pretrained_models.html
    """
    # Data Locations.
    models_folder = 'doc_models'

    # Bert Models used.
    models_dict = {
        # English Models.
        'all-mpnet-base-v2': {
            'type': 'bert-english',
            'max_seq_length': 384,
            'dimensions': 768,
            'performance': 63.30,
            'speed': 2_800,
            'size': 420,
        },
        'all-MiniLM-L12-v2': {
            'type': 'bert-english',
            'max_seq_length': 256,
            'dimensions': 384,
            'performance': 59.76,
            'speed': 7_500,
            'size': 120,
        },
        'all-MiniLM-L6-v2': {
            'type': 'bert-english',
            'max_seq_length': 256,
            'dimensions': 384,
            'performance': 58.80,
            'speed': 14_200,
            'size': 80,
        },
        'paraphrase-MiniLM-L3-v2': {
            'type': 'bert-english',
            'max_seq_length': 128,
            'dimensions': 384,
            'performance': 50.74,
            'speed': 19_000,
            'size': 61,
        },
        # GloVe Model.
        'average_word_embeddings_glove.6B.300d': {
            'type': 'glove',
            'max_seq_length': -1,
            'dimensions': 300,
            'performance': 36.25,
            'speed': 34_000,
            'size': 420,
        },
        # Multilingual Models (50+ languages).
        'paraphrase-multilingual-mpnet-base-v2': {
            'type': 'bert-multilingual',
            'max_seq_length': 128,
            'dimensions': 768,
            'performance': 53.75,
            'speed': 2_500,
            'size': 970,
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'type': 'bert-multilingual',
            'max_seq_length': 128,
            'dimensions': 384,
            'performance': 51.72,
            'speed': 7_500,
            'size': 420,
        }
    }

    def __init__(self, model_name='', show_progress=False):
        """
        Load or Download the requested Sentence-Bert 'model_name'. The model
        will be used to get the embeddings words and documents. By default, it
        loads the fastest SBERT model 'paraphrase-MiniLM-L3-v2'.

        Args:
            model_name: A String with the name of model.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check if a model name was provided.
        if not model_name:
            model_name = 'all-MiniLM-L6-v2'
        # Check the model we are using is supported by the class.
        if model_name not in self.models_dict:
            raise NameError("SBertModel() does not support the requested model.")

        # Create models' folder if it doesn't exist.
        if not isdir(self.models_folder):
            mkdir(self.models_folder)

        # Load or Download Model.
        model_path = join(self.models_folder, model_name)
        if not isdir(model_path):
            if show_progress:
                progress_msg(f"Downloading Sentence-Bert model <{model_name}>...")
            # Download model.
            model = SentenceTransformer(f'sentence-transformers/{model_name}')
            # Save model locally.
            model.save(model_path, model_name, create_model_card=True)
            if show_progress:
                progress_msg(f"SBERT model <{model_name}> downloaded and saved locally.")
        else:
            # The model is available locally.
            model = SentenceTransformer(model_path)
        if show_progress:
            progress_msg(f"SBERT model <{model_name}> loaded successfully.")
            model_dict = self.models_dict[model_name]
            progress_msg(f"  Speed: {big_number(model_dict['speed'])}")
            progress_msg(f"  Accuracy: {model_dict['performance']}")
            progress_msg(f"  Max Seq Length: {model_dict['max_seq_length']} tokens")

        # Save model and its name.
        self.model = model
        self.model_name = model_name

    def text_vector(self, text: str):
        """
        Transform a text (word, sentence or paragraph) into its vector
        representation.

        Args:
            text: String with the content of the 'text'.
        Returns:
            Numpy.ndarray with the embedding of the 'text'.
        """
        embed = self.model.encode(text)
        return embed

    def text_list_vectors(self, text_list: list):
        """
        Get the vector representation of a list of texts.

        Args:
            text_list: List[str] containing the content of the texts.
        Returns:
            List[Numpy.ndarray] with the embeddings of the texts.
        """
        embeds_list = self.model.encode(text_list)
        return embeds_list


def load_all_models():
    """
    Test loading all the supported Bert Models to make sure that they are saved
    locally and work properly.
    """
    # Keep track of the runtime of the method.
    tracker = TimeKeeper()

    # Get the names of the models.
    supported_models = list(SBertModel.models_dict)

    # Test Loading each model.
    for model_name in supported_models:
        print("-------------------------------------------------------")
        print(f"\nLoading Model <{model_name}>:")
        new_model = SBertModel(model_name=model_name, show_progress=True)
        print(f"The model {new_model.model_name} is ready.")
        print("Done.")
        print(f"[{tracker.formatted_runtime()}]\n")


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()
    # Terminal Arguments.
    args = sys.argv

    # Creating Model.
    print("\nCreating Sentence-Bert Model...")
    my_model = SBertModel(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # # Test Word Embeddings:
    # while True:
    #     word_input = input("\nType a word (q/quit to close): ")
    #     word_input = word_input.strip().lower()
    #     if word_input in {'', 'q', 'quit'}:
    #         break
    #     word_embed = my_model.text_vector(word_input)
    #     print(f"\nThe embedding of <{word_input}>:")
    #     print(word_embed)
    #     print(f"Type: {type(word_embed)}")

    # Check Similarities between words.
    print("\nTesting word similarities (To close use [q/quit]):")
    quit_words = {'q', 'quit'}
    while True:
        word1 = input("\nType the first word: ")
        if word1 in quit_words:
            break
        word2 = input("Type the second word: ")
        if word2 in quit_words:
            break
        sim_words = cos_sim(my_model.text_vector(word1), my_model.text_vector(word2))
        print("\nWords similarity:")
        print(sim_words)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
