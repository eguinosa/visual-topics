# Gelin Eguinosa Rosique
# 2022

import json
from os import mkdir
from os.path import isdir, isfile, join
from shutil import rmtree

from topic_corpus import TopicCorpus
from model_manager import ModelManager
from doc_tokenizer import DocTokenizer
from util_funcs import dict_list2ndarray, dict_ndarray2list
from extra_funcs import progress_bar, progress_msg, big_number

# Testing Imports.
import sys
from sample_manager import SampleManager
from util_funcs import top_n
from time_keeper import TimeKeeper


class Vocabulary:
    """
    Manage the vocabulary of the corpus creating an id for each word, saving
    the word embeddings and the documents' vocabulary, so we can see to which
    documents each word belongs to. It also has tools to calculate the
    frequencies of the words in the documents and the corpus.

    The vocabulary files are meant to be saved inside a given Topic Model's
    Folder.
    """
    # Class variables filenames.
    vocab_folder_name = 'vocabulary_files'
    vocab_index_file = 'vocabulary_index.json'
    word_embeds_file = 'word2embed.json'

    def __init__(self, corpus: TopicCorpus = None, model: ModelManager = None,
                 _load_vocab=False, _vocab_dir_path='', show_progress=False):
        """
        Analyze the given corpus to extract the words in the vocabulary of the
        documents, get their embeddings and find their frequencies inside each
        document and the entire corpus.

        Args:
            corpus: TopicCorpus instance representing the corpus used to
                create the Topic Model to which this vocabulary will belong.
            model: ModelManager instance used to create the embeddings of the
                words in the vocabulary.
            _load_vocab: Bool indicating whether we have to load the vocabulary
                from a local directory.
            _vocab_dir_path: Directory where the vocabulary files are stored.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check if we have to load a previously saved vocabulary.
        if _load_vocab:
            # Check if we have a valid '_vocab_dir_path':
            if not _vocab_dir_path:
                raise Exception("No Path was provided to load the Vocabulary.")
            if not isdir(_vocab_dir_path):
                raise Exception(f"The provided path <({_vocab_dir_path})> is not a folder.")

            # Load class Index.
            if show_progress:
                progress_msg("Loading the Vocabulary Index...")
            index_path = join(_vocab_dir_path, self.vocab_index_file)
            if not isfile(index_path):
                raise FileNotFoundError("The Vocabulary index file does not exist.")
            with open(index_path, 'r') as f:
                vocab_index = json.load(f)
            # Get the Index variables.
            corpus_length = vocab_index['corpus_length']
            corpus_freqs = vocab_index['corpus_freqs']
            docs_lengths = vocab_index['docs_lengths']
            docs_freqs = vocab_index['docs_freqs']

            # Load Word Embeddings.
            if show_progress:
                progress_msg("Loading Vocabulary Embeddings...")
            word_embeds_path = join(_vocab_dir_path, self.word_embeds_file)
            if not isfile(word_embeds_path):
                raise FileNotFoundError("The Vocabulary's Embeddings file does not exist.")
            with open(word_embeds_path, 'r') as f:
                word_embeds_index = json.load(f)
            if show_progress:
                progress_msg("Transforming Word's Embeddings back to Numpy.ndarray...")
            # Transform embeddings back to Numpy.ndarray.
            word_embeds = dict_list2ndarray(word_embeds_index, show_progress=show_progress)
            # Done loading Vocabulary.
            if show_progress:
                progress_msg("Vocabulary Loaded.")
        else:
            # Check we have a corpus and a model.
            if not corpus or not model:
                raise Exception("We need the corpus and model to create the Vocabulary.")

            # Variables of the Vocabulary for the given 'corpus'.
            corpus_length = 0
            corpus_freqs = {}
            docs_lengths = {}
            docs_freqs = {}
            word_embeds = {}

            # Progress Bar Variables.
            count = 0
            total = len(corpus)
            # Create tokenizer to  process the content of the documents.
            tokenizer = DocTokenizer()
            # Extract the title & abstract of the papers and tokenize them.
            if show_progress:
                progress_msg("Processing the Documents in the corpus...")
            for doc_id in corpus.doc_ids:
                doc_content = corpus.doc_title_abstract(doc_id)
                doc_tokens = tokenizer.vocab_tokenizer(doc_content)
                # Create embeddings for the new words.
                new_words = [token for token in doc_tokens if token not in word_embeds]
                new_embeds = model.word_list_embeds(new_words)
                for new_word, new_embed in zip(new_words, new_embeds):
                    word_embeds[new_word] = new_embed
                # Update corpus data and create frequencies for the document.
                doc_word_count = {}
                for token in doc_tokens:
                    # Update corpus data.
                    corpus_length += 1
                    if token in corpus_freqs:
                        corpus_freqs[token] += 1
                    else:
                        corpus_freqs[token] = 1
                    # Document frequencies.
                    if token in doc_word_count:
                        doc_word_count[token] += 1
                    else:
                        doc_word_count[token] = 1
                # Save the Document's length and frequencies.
                docs_lengths[doc_id] = len(doc_tokens)
                docs_freqs[doc_id] = doc_word_count
                # Progress.
                if show_progress:
                    count += 1
                    progress_bar(count, total)
            # Done creating Vocabulary.
            if show_progress:
                progress_msg("Vocabulary Created.")

        # Save the Vocabulary Info.
        self.corpus_length = corpus_length
        self.corpus_freqs = corpus_freqs
        self.docs_lengths = docs_lengths
        self.docs_freqs = docs_freqs
        self.word_embeds = word_embeds

        # Vocabulary Statistics.
        if show_progress:
            progress_msg(f"  -> {big_number(len(self.corpus_docs))} documents analyzed.")
            progress_msg(f"  -> {big_number(self.corpus_length)} word tokens processed.")
            progress_msg(f"  -> {big_number(len(self.corpus_words))} unique terms found.")

    @property
    def corpus_words(self):
        """
        Get the words in the vocabulary of the corpus.

        Returns: List[str] with the words in the vocabulary.
        """
        vocab_words = list(self.corpus_freqs)
        return vocab_words

    @property
    def corpus_docs(self):
        """
        Get the documents used to create this vocabulary.

        Returns: List[str] with the IDs of the documents in the corpus.
        """
        doc_ids = list(self.docs_freqs)
        return doc_ids

    def doc_words(self, doc_id: str):
        """
        The word in the vocabulary of the document 'doc_id'.

        Returns: List[str] with the words of the document.
        """
        doc_freqs = self.docs_freqs[doc_id]
        doc_vocab = list(doc_freqs)
        return doc_vocab

    def save(self, topic_dir_path: str, show_progress=False):
        """
        Save the Vocabulary data.

        Args:
            topic_dir_path: String with the path to the directory where the
                Topic Model is (or will be) saved.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check if we received a valid directory path.
        if not isdir(topic_dir_path):
            raise NotADirectoryError("The provided directory path is not a folder.")
        # Create path for the Vocabulary folder and check it is empty.
        vocab_folder_path = join(topic_dir_path, self.vocab_folder_name)
        if isdir(vocab_folder_path):
            rmtree(vocab_folder_path)
        mkdir(vocab_folder_path)

        # Save Vocabulary Index.
        if show_progress:
            progress_msg("Saving Vocabulary Index File...")
        vocab_index = {
            'corpus_length': self.corpus_length,
            'corpus_freqs': self.corpus_freqs,
            'docs_lengths': self.docs_lengths,
            'docs_freqs': self.docs_freqs,
        }
        vocab_index_path = join(vocab_folder_path, self.vocab_index_file)
        with open(vocab_index_path, 'w') as f:
            json.dump(vocab_index, f)

        # Transform word embeddings from Numpy.nd to List.
        if show_progress:
            progress_msg("Transforming Vocabulary Word's Embeddings to List[float]...")
        word_embeds_index = dict_ndarray2list(self.word_embeds, show_progress=show_progress)
        # Save Embeddings.
        if show_progress:
            progress_msg("Saving Vocabulary Embedding's file...")
        word_embeds_path = join(vocab_folder_path, self.word_embeds_file)
        with open(word_embeds_path, 'w') as f:
            json.dump(word_embeds_index, f)

    @classmethod
    def load(cls, topic_dir_path: str, show_progress=False):
        """
        Load a Vocabulary given the folder of the Topic Model who saved it.

        Args:
            topic_dir_path: String with the path to the directory where the
                Vocabulary's Topic Model is saved.
            show_progress: Bool representing whether we show the progress of
                the method or not.
        """
        # Check we have a valid topic path.
        if not isdir(topic_dir_path):
            raise NotADirectoryError("There is no directory in the provided path.")
        # Create and Check path to the Vocabulary folder.
        vocab_folder_path = join(topic_dir_path, cls.vocab_folder_name)
        if not isdir(vocab_folder_path):
            raise NotADirectoryError("The provided Topic Path has no Vocabulary saved.")

        # Load the Vocabulary class.
        vocab = cls(
            _load_vocab=True, _vocab_dir_path=vocab_folder_path, show_progress=show_progress
        )
        return vocab


if __name__ == '__main__':
    # Record Program Runtime.
    stopwatch = TimeKeeper()
    # Terminal Arguments.
    args = sys.argv

    # Create corpus.
    my_docs_num = 50
    print(f"\nCreating Corpus Sample of {big_number(my_docs_num)} documents...")
    my_sample = SampleManager(sample_size=my_docs_num, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Creating Text Model.
    print("\nCreating Text Model for the vocabulary embeddings...")
    my_model = ModelManager(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Create Vocabulary.
    print("\nCreating Vocabulary using the created corpus and model...")
    my_vocab = Vocabulary(corpus=my_sample, model=my_model, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Get the Most Frequent Words in the Corpus.
    my_top_num = 10
    print(f"\nFinding the Top {my_top_num} most frequent words in the corpus:")
    my_top_words = top_n(my_vocab.corpus_freqs.items(), n=my_top_num)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print(f"\nTop {my_top_num} Most Frequent Words:")
    for a_word, a_count in my_top_words:
        print(f"  {a_word} -> {a_count}")

    # Get the Biggest Documents in the Corpus.
    my_top_num = 10
    print(f"\nThe Top Biggest {my_top_num} documents")
    my_big_docs = top_n(my_vocab.docs_lengths.items(), n=my_top_num)
    for a_doc_id, a_size in my_big_docs:
        print(f"  Doc <{a_doc_id}>: {a_size} tokens")

    # Test Saving & Loading the Vocabulary.

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
