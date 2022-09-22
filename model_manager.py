# Gelin Eguinosa
# 2022

import sys
from sbert_model import SBertModel

# Testing Imports
from util_funcs import closest_vector
from time_keeper import TimeKeeper


class ModelManager:
    """
    Class to load and control the document models used with the Topic Models.
    """
    # Default Model.
    default_model = 'sbert_fast'

    # Supported Models.
    models_dict = {
        'sbert_best': {
            'name': 'all-mpnet-base-v2',
            'type': 'sbert',
        },
        'sbert_fast': {
            'name': 'all-MiniLM-L6-v2',
            'type': 'sbert',
        },
        'multilingual_best': {
            'name': 'paraphrase-multilingual-mpnet-base-v2',
            'type': 'sbert',
        },
        'multilingual_fast': {
            'name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'type': 'sbert',
        },
    }

    def __init__(self, model_name='', show_progress=False):
        """
        Load and store the document model 'model_name'. The 'model_name' needs
        to be one of the supported models inside 'self.models_dict'.

        Args:
            model_name: String with the ModelManager ID indicating which model
                we are loading.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check we have a model name.
        if not model_name:
            model_name = self.default_model
        # Check we support the model name.
        if model_name not in self.models_dict:
            raise NameError("ModelManager() does not support the requested model.")

        # Get model Info and load the model.
        model_info = self.models_dict[model_name]
        if model_info['type'] == 'sbert':
            model = SBertModel(model_name=model_info['name'], show_progress=show_progress)
        else:
            raise Exception(f"Document Models <{model_info['type']}> not implemented yet.")

        # Save Model & Info.
        self.model = model
        self.model_name = model_name
        self.model_type = model_info['type']
        self.model_full_name = model_info['name']

    def word_embed(self, word: str):
        """
        Get the vector representation of the token 'word'.

        Args:
            word: String with the text of a token.
        Returns:
            Numpy.ndarray with the vector of the word.
        """
        if self.model_type == 'sbert':
            embed = self.model.text_embed(word)
        else:
            raise Exception(f"Document Models <{self.model_type}> not implemented yet.")

        # The vector representation of the word.
        return embed

    def word_list_embeds(self, words_list: list):
        """
        Get the vector representation of the list of tokens 'words'.

        Args:
            words_list: List[str] containing the texts of a list of tokens.
        Returns:
            List[Numpy.ndarray] with the vectors of the words.
        """
        if self.model_type == 'sbert':
            embeds_list = self.model.text_list_embeds(words_list)
        else:
            raise Exception(f"Document Models <{self.model_type}> not implemented yet.")

        # List of embeddings.
        return embeds_list

    def doc_embed(self, doc: str):
        """
        Get the vector representation of a document 'doc'.

        Args:
            doc: String with the text of the document.
        Returns:
            Numpy.ndarray with the vector of the document.
        """
        if self.model_type == 'sbert':
            embed = self.model.text_embed(doc)
        else:
            raise Exception(f"Document Models <{self.model_type}> not implemented yet.")

        # The vector representation of the document.
        return embed

    def doc_list_embeds(self, docs_list: list):
        """
        Get the vector representation of the documents in 'docs_list'.

        Args:
            'docs_list': List[str] with the texts of the documents.
        Returns:
            List[Numpy.ndarray] with the vectors of the documents.
        """
        if self.model_type == 'sbert':
            embeds_list = self.model.text_list_embeds(docs_list)
        else:
            raise Exception(f"Document Models <{self.model_type}> not implemented yet.")

        # List of embeddings.
        return embeds_list


if __name__ == '__main__':
    # Record the Runtime of the Program
    _stopwatch = TimeKeeper()
    # Terminal Arguments.
    _args = sys.argv

    # Loading Model Manager.
    _model_id = 'multilingual_fast'
    print(f"\nLoading the model in ModelManager <{_model_id}>...")
    _model = ModelManager(model_name=_model_id, show_progress=True)
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")

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
        _word_embed = _model.word_embed(_word)
        _embeds_list = _model.word_list_embeds(_word_list)
        _embeds_dict = dict(zip(_word_list, _embeds_list))
        # Get the closest word.
        _closest_word, _sim = closest_vector(_word_embed, _embeds_dict)
        # Report Word and Similarity.
        print(f"The closest word to <{_word}>:")
        print(f" Word -> {_closest_word}")
        print(f" Sim  -> {_sim}")

    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
