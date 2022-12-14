# Gelin Eguinosa Rosique
# 2022

import json
import math
from os import mkdir
from os.path import isdir, isfile, join
from shutil import rmtree
from collections import defaultdict

from topic_corpus import TopicCorpus
from model_manager import ModelManager
from doc_tokenizer import DocTokenizer
from util_funcs import dict_list2ndarray, dict_ndarray2list
from extra_funcs import progress_bar, progress_msg, big_number

# Testing Imports.
import sys
from pprint import pprint
from sample_manager import SampleManager
from util_funcs import find_top_n
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
            word_docs_count = vocab_index['word_docs_count']
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
            word_docs = {}
            corpus_freqs = {}
            docs_lengths = {}
            docs_freqs = {}
            word_embeds = {}
            # To Save the Different Casing Variations of the Token in the texts.
            token_variations = {}

            # Progress Bar Variables.
            count = 0
            total = len(corpus) + 3  # Common Token (1), Update Dict Keys (2,3)
            # Create tokenizer to  process the content of the documents.
            tokenizer = DocTokenizer()
            # Extract the title & abstract of the papers and tokenize them.
            if show_progress:
                progress_msg("Processing the Documents in the corpus...")
            for doc_id in corpus.doc_ids:
                # Get the tokens of the document.
                doc_content = corpus.doc_title_abstract(doc_id)
                doc_tokens = tokenizer.vocab_tokenizer(doc_content)
                # Analyze the Tokens to save new words and token's variations.
                # Update corpus data and create frequencies for the document.
                new_words = []
                doc_word_count = {}
                for token in doc_tokens:
                    # Use token in lower form.
                    token_lower = token.lower()
                    # Save new Words.
                    if token_lower not in word_embeds:
                        new_words.append(token_lower)
                    # Update the Variations of the Token. (Upper, Lower, or mixed).
                    if token_lower in token_variations:
                        token_reps = token_variations[token_lower]
                        token_reps[token] += 1
                    else:
                        token_reps = defaultdict(int)
                        token_reps[token] += 1
                        token_variations[token_lower] = token_reps
                    # Update Word Documents.
                    if token_lower in word_docs:
                        word_docs[token_lower].add(doc_id)
                    else:
                        word_docs[token_lower] = set(doc_id)
                    # Update corpus data.
                    corpus_length += 1
                    if token_lower in corpus_freqs:
                        corpus_freqs[token_lower] += 1
                    else:
                        corpus_freqs[token_lower] = 1
                    # Update Document Frequencies.
                    if token_lower in doc_word_count:
                        doc_word_count[token_lower] += 1
                    else:
                        doc_word_count[token_lower] = 1
                    # Update
                # Create embeddings for the new words.
                new_embeds = model.word_list_embeds(new_words)
                for new_word, new_embed in zip(new_words, new_embeds):
                    word_embeds[new_word] = new_embed
                # Save the Document's length and frequencies.
                docs_lengths[doc_id] = len(doc_tokens)
                docs_freqs[doc_id] = doc_word_count
                # Progress.
                if show_progress:
                    count += 1
                    progress_bar(count, total)

            # Get all the tokens found in lower case.
            corpus_tokens_lower = list(token_variations.keys())
            # Get the most common representation of the tokens in the corpus.
            for token_lower in corpus_tokens_lower:
                # Token's Variations in the corpus.
                token_reps = token_variations[token_lower]
                # Token with only one representation in the corpus.
                if len(token_reps) == 1:
                    common_rep = list(token_reps)[0]
                # Token with multiple representations.
                else:
                    # Use the most common representation.
                    common_rep = max(
                        token_reps.keys(), key=lambda d_key: token_reps[d_key]
                    )
                token_variations[token_lower] = common_rep
            if show_progress:
                count += 1
                progress_bar(count, total)
            # Use Most Common Representation in Embeddings and Corpus Frequencies.
            # Create dictionary with the number of document where the words appear.
            word_docs_count = {}
            for token_lower in corpus_tokens_lower:
                # Get new Representation.
                new_token_rep = token_variations[token_lower]
                # Get number of Docs where the Word appears.
                doc_set = word_docs[token_lower]
                word_docs_count[new_token_rep] = len(doc_set)
                # Check if the token has the same most common representation.
                if token_lower == new_token_rep:
                    continue
                # Update Keys of Corpus Frequencies.
                freq_value = corpus_freqs[token_lower]
                del corpus_freqs[token_lower]
                corpus_freqs[new_token_rep] = freq_value
                # Update Keys of Word Embeddings.
                embed_value = word_embeds[token_lower]
                del word_embeds[token_lower]
                word_embeds[new_token_rep] = embed_value
            if show_progress:
                count += 1
                progress_bar(count, total)
            # Update to Most Common Representation in Document Frequencies.
            for doc_word_count in docs_freqs.values():
                # Iterate through the words in the document's frequencies.
                doc_count_keys = list(doc_word_count.keys())
                for token_lower in doc_count_keys:
                    # Skip if the common representation is token_lower.
                    if token_lower == token_variations[token_lower]:
                        continue
                    # Update Key of the Document's frequencies.
                    new_token_rep = token_variations[token_lower]
                    word_count = doc_word_count[token_lower]
                    del doc_word_count[token_lower]
                    doc_word_count[new_token_rep] = word_count
            if show_progress:
                count += 1
                progress_bar(count, total)
            # Done creating Vocabulary.
            if show_progress:
                progress_msg("Vocabulary Created.")

        # Save the Vocabulary Info.
        self.corpus_length = corpus_length
        self.word_docs_count = word_docs_count
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

    def pwi_tf_idf(self, word: str, doc_id: str):
        """
        Get the tf-idf Probability-Weighted amount of Information (PWI) between
        the 'word' and the document 'doc_id'.

        Args:
            word: String with the text of the word.
            doc_id: String with the ID of the document.
        Returns:
            Float with the PWI tf-idf of the word and the document.
        """
        # Doc Frequencies.
        doc_freqs = self.docs_freqs[doc_id]
        # Check the word is in the Document.
        if word not in doc_freqs:
            return 0
        # Get the values needed to calculate the PWI.
        word_doc_freq = doc_freqs[word]
        corpus_length = self.corpus_length
        word_doc_count = self.word_docs_count[word]
        corpus_size = len(self.docs_lengths)
        # Calculate the PWI tf-idf.
        pwi_result = (
            (word_doc_freq/corpus_length) * math.log(corpus_size/word_doc_count)
        )
        return pwi_result

    def pwi_exact(self, word: str, doc_id: str):
        """
        Get the exact Probability-Weighted amount of Information (PWI) between
        the 'word' and the document 'doc_id'.

        Args:
            word: String with the text of the word.
            doc_id: String with the ID of the document.
        Returns:
            Float with the PWI exact of the word and the document.
        """
        # Doc Frequencies.
        doc_freqs = self.docs_freqs[doc_id]
        # Check the word is in the Document.
        if word not in doc_freqs:
            return 0
        # Get the values of the variables to calculate PWI exact.
        word_doc_freq = doc_freqs[word]
        corpus_length = self.corpus_length
        word_corpus_freq = self.corpus_freqs[word]
        doc_length = self.docs_lengths[doc_id]
        # Calculate PWI exact.
        pwi_result = (
            (word_doc_freq/corpus_length) *
            math.log(
                (word_doc_freq * corpus_length) / (word_corpus_freq * doc_length)
            )
        )
        return pwi_result

    def word_pwi_tf_idf(self, word: str, doc_ids: list):
        """
        Get the tf-idf Probability-Weighted amount of Information (PWI-tf-idf)
        between the 'word' and the list of documents 'doc_ids'.

        Args:
            word: String with the text of the word.
            doc_ids: List[str] with the IDs of the documents.
        Returns:
            Float with the PWI-tf-idf of the word and the documents.
        """
        # Get the Number of documents in the corpus.
        corpus_size = len(self.docs_lengths)
        # Get the sum of the word frequency in the documents.
        if len(doc_ids) == corpus_size:
            # The Docs are the whole corpus. Get the corpus frequency instead.
            word_docs_freq = self.corpus_freqs[word]
        else:
            # Use the doc_ids for the word total frequency.
            word_docs_freq = sum(
                doc_freqs[word] for doc_id in doc_ids
                if word in (doc_freqs := self.docs_freqs[doc_id])
            )
        # Create the rest of the variables for the formula.
        corpus_length = self.corpus_length
        word_doc_count = self.word_docs_count[word]
        # Calculate the PWI-tf-idf.
        pwi_result = (
                (word_docs_freq / corpus_length) * math.log(corpus_size/word_doc_count)
        )
        return pwi_result

    def word_pwi_exact(self, word: str, doc_ids: list):
        """
        Get the exact Probability-Weighted amount of Information (PWI-exact)
        between the 'word' and the list of documents 'doc_ids'.

        Args:
            word: String with the text of the word.
            doc_ids: List[str] with the IDs of the documents.
        Returns:
            Float with the PWI-exact of the word and the documents.
        """
        # Get corpus variables.
        word_corpus_freq = self.corpus_freqs[word]
        corpus_length = self.corpus_length
        # Calculate the summation in the formula.
        summation = 0
        for doc_id in doc_ids:
            # Get Frequencies of words in the current Document.
            doc_freqs = self.docs_freqs[doc_id]
            # Check the word is in the Document.
            if word not in doc_freqs:
                continue
            # Variables of the summation.
            word_doc_freq = doc_freqs[word]
            doc_length = self.docs_lengths[doc_id]
            # Calculate Iteration.
            summand = (
                word_doc_freq *
                math.log(
                    (corpus_length * word_doc_freq) / (word_corpus_freq * doc_length)
                )
            )
            # Add iteration to summation.
            summation += summand
        # Divide summation with the corpus frequency.
        pwi_result = summation / corpus_length
        return pwi_result

    def most_common_term(self):
        """
        Find the word(s) with the highest frequency in the corpus.
        """
        # Extract the words and their frequencies.
        words_freqs = list(self.corpus_freqs.items())
        # Create Reference values.
        first_word, first_freq = words_freqs[0]
        top_words = [first_word]
        max_freq = first_freq
        for new_word, new_freq in words_freqs[1:]:
            # Check - New Max.
            if new_freq > max_freq:
                top_words = [new_word]
                max_freq = new_freq
            # Check - Same Max Frequency.
            elif new_freq == max_freq:
                top_words.append(new_word)
        # The Top Frequency Words.
        return top_words, max_freq

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
            'word_docs_count': self.word_docs_count,
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

    @classmethod
    def vocab_saved(cls, topic_dir_path: str):
        """
        Check if a Vocabulary is saved in the given directory.

        Args:
            topic_dir_path: String with the path to the directory where the
                Vocabulary is supposed to be saved.
        Returns:
            Bool indicating if a Vocabulary was saved in this folder or not.
        """
        # Check we have a valid topic path.
        if not isdir(topic_dir_path):
            return False
        # Check Vocabulary Path Folder.
        vocab_folder_path = join(topic_dir_path, cls.vocab_folder_name)
        if not isdir(vocab_folder_path):
            return False
        # Check Vocabulary Index.
        vocab_index_path = join(vocab_folder_path, cls.vocab_index_file)
        if not isfile(vocab_index_path):
            return False
        # Check Vocabulary Word Embeddings.
        word_embeds_path = join(vocab_folder_path, cls.word_embeds_file)
        if not isfile(word_embeds_path):
            return False
        # All Good: Folder and files present.
        return True


if __name__ == '__main__':
    # Record Program Runtime.
    _stopwatch = TimeKeeper()
    # Terminal Arguments.
    _args = sys.argv

    # Create corpus.
    # ------------------------------------------------
    _docs_num = 50
    print(f"\nCreating Corpus Sample of {big_number(_docs_num)} documents...")
    _sample = SampleManager(sample_size=_docs_num, show_progress=True)
    # ------------------------------------------------
    # _sample_id = '100_docs'
    # print(f"\nLoading Corpus Sample with ID <{_sample_id}>...")
    # _sample = SampleManager.load(sample_id=_sample_id)
    # ------------------------------------------------
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")

    print(f"\n{big_number(len(_sample))} documents loaded.")

    # Creating Text Model.
    print("\nCreating Text Model for the vocabulary embeddings...")
    _model = ModelManager(show_progress=True)
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")

    # Create Vocabulary.
    print("\nCreating Vocabulary using the created corpus and model...")
    _vocab = Vocabulary(corpus=_sample, model=_model, show_progress=True)
    print("Done.")
    print(f"[{_stopwatch.formatted_runtime()}]")

    # # --- Test Saving Vocabulary ---
    # print("\nSaving Vocabulary...")
    # _vocab.save(topic_dir_path='temp_data', show_progress=True)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # # --- Test Loading the Vocabulary ---
    # _location = 'temp_data'
    # print(f"\nChecking if a Vocabulary is saved in the path: '{_location}'")
    # _is_saved = Vocabulary.vocab_saved(topic_dir_path=_location)
    # if _is_saved:
    #     print("Yes, there is a Vocabulary in the folder.")
    # else:
    #     print("No, there is no Vocabulary saved in that folder.")
    # # # ----------------------------------
    # print("\nLoading Saved Vocabulary...")
    # _vocab = Vocabulary.load(topic_dir_path='temp_data', show_progress=True)
    # print("Done.")
    # print(f"[{_stopwatch.formatted_runtime()}]")

    # Get the Most Frequent Words in the Corpus.
    _top_num = 10
    print(f"\nFinding the Top {_top_num} most frequent words in the corpus:")
    _top_words = find_top_n(_vocab.corpus_freqs.items(), n=_top_num)
    print(f"Top {_top_num} Most Frequent Words:")
    for _word, _count in _top_words:
        print(f"  {_word} -> {_count}")

    # -- Get the most informative words in the corpus --
    print("\nGetting the TF-IDF most informative words in the Corpus...")
    _words_tf_idf = [
        (word, _vocab.word_pwi_tf_idf(word, _vocab.corpus_docs))
        for word in _vocab.corpus_words
    ]
    _info_words = find_top_n(_words_tf_idf, n=10)
    print("Top 10 most informative Words (TF-IDF):")
    pprint(_info_words)
    # ------------------------------------------------
    print("\nGetting the PWI-exact most informative words in the Corpus...")
    _words_exact = [
        (word, _vocab.word_pwi_exact(word, _vocab.corpus_docs))
        for word in _vocab.corpus_words
    ]
    _info_words = find_top_n(_words_exact, n=10)
    print("Top 10 most informative Words (PWI-exact):")
    pprint(_info_words)

    # ------------------------------------------------
    # # Get the Biggest Documents in the Corpus.
    # _top_num = 10
    # print(f"\nThe Top Biggest {_top_num} documents")
    # _big_docs = find_top_n(_vocab.docs_lengths.items(), n=_top_num)
    # for _doc_id, _size in _big_docs:
    #     print(f"  Doc <{_doc_id}>: {_size} tokens")

    # # -- Test PWI-tf-idf & PWI-exact --
    # # Get content of the 15th smallest Document.
    # _small_docs = find_top_n(_vocab.docs_lengths.items(), n=15, top_max=False)
    # _least_big_id, _ = _small_docs[-2]
    # _least_big_content = _sample.doc_title_abstract(_least_big_id)
    # print(f"\nContent of the Document <{_least_big_id}>:")
    # pprint(_least_big_content, width=80)
    # # ------------------------------------------------
    # # Get the Mutual Information between the Doc and its tokens.
    # _words_tf_idf = [
    #     (word, _vocab.pwi_tf_idf(word, _least_big_id))
    #     for word in _vocab.doc_words(_least_big_id)
    # ]
    # _info_words = find_top_n(_words_tf_idf, n=10)
    # print("\nThe Top 10 most informative words with PWI-tf-idf:")
    # pprint(_info_words)
    # # ------------------------------------------------
    # _words_exact = [
    #     (word, _vocab.pwi_exact(word, _least_big_id))
    #     for word in _vocab.doc_words(_least_big_id)
    # ]
    # _info_words = find_top_n(_words_exact, n=10)
    # print("\nThe Top 10 most informative words with PWI-exact:")
    # pprint(_info_words)

    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
