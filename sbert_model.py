# Gelin Eguinosa
# 2022

import sys
from os import mkdir
from os.path import isdir, join
from sentence_transformers import SentenceTransformer

from extra_funcs import progress_msg, big_number
from time_keeper import TimeKeeper

# Test Imports
from util_funcs import closest_vector, cosine_sim


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

    def text_embed(self, text: str):
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

    def text_list_embeds(self, text_list: list):
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
    _stopwatch = TimeKeeper()
    # Terminal Arguments.
    _args = sys.argv

    # Creating Model.
    print("\nCreating Sentence-Bert Model...")
    _model = SBertModel(show_progress=True)
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")

    # Test Word Embeddings:
    while True:
        _input = input("\nType a word (q/quit to close): ")
        _input = _input.strip().lower()
        if _input in {'', 'q', 'quit'}:
            break
        _embed = _model.text_embed(_input)
        print(f"\nThe embedding of <{_input}>:")
        print(_embed)
        print(f"Type: {type(_embed)}")

    # Check Similarities between words.
    print("\nTesting word similarities (To close use [q/quit]):")
    while True:
        _word1 = input("\nType the first word: ")
        if _word1 in {'', 'q', 'quit'}:
            break
        _word2 = input("Type the second word: ")
        if _word2 in {'', 'q', 'quit'}:
            break
        _sim_words = cosine_sim(_model.text_embed(_word1), _model.text_embed(_word2))
        print("\nWords similarity:")
        print(_sim_words)

    # Find most similar word.
    print("\nGiven a word, and a list of words, select the word most similar.")
    print("(To close use [q/quit])")
    while True:
        _word = input("\nType the search word: ")
        if _word in {'', 'q', 'quit'}:
            break
        _list = input("Type the list of words to search on. (Use commas to "
                      "separate the words)\n -> ")
        if _list in {'', 'q', 'quit'}:
            break
        _word_list = [a_word.strip() for a_word in _list.split(',')]
        _word_embed = _model.text_embed(_word)
        _embeds_list = _model.text_list_embeds(_word_list)
        _embeds_dict = dict(zip(_word_list, _embeds_list))
        # Get the closest word.
        _closest_word, _sim = closest_vector(_word_embed, _embeds_dict)
        # Report Word and Similarity.
        print(f"The closest word to <{_word}>:")
        print(f" Word -> {_closest_word}")
        print(f" Sim  -> {_sim}")

    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
