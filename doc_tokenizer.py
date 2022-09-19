# Gelin Eguinosa Rosique
# 2022

import sys
import spacy
from os import mkdir
from os.path import isdir, join
from spacy.util import compile_infix_regex
from spacy.tokens.token import Token
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS

# Testing Imports.
from pprint import pprint
from sample_manager import SampleManager
from time_keeper import TimeKeeper


class DocTokenizer:
    """
    Class to tokenize the documents in the corpus using spacy.
    """
    # Class Data Locations.
    spacy_folder = 'spacy_models'
    default_model = 'en_core_web_sm'
    # Supported Spacy Models.
    models_allowed = {'en_core_web_sm'}

    def __init__(self, spacy_model='', hyphens=False):
        """
        Load the spacy model used for the tokenization of the documents in the
        corpus, if no 'spacy_model' is provided, loads the 'default_model'.

        If the value of 'hyphens' is true. The Infix Pattern of the tokenizer
        will be updater to accept hyphens (-) inside words instead of splitting
        them.

        Args:
            spacy_model: String with the name of the spacy model we want to work
                with.
            hyphens: Bool indicating if the model should accept tokens with
                hyphenation (True), or split the words by their hyphens (False).
        """
        # Create the Spacy's Models folder if it doesn't exist.
        if not isdir(self.spacy_folder):
            mkdir(self.spacy_folder)

        # See if we need to load a custom model.
        if spacy_model:
            # Check we are loading one of the supported models.
            if spacy_model not in self.models_allowed:
                raise Exception(f"The Tokenizer doesn't support the spacy model <{spacy_model}>.")
            # Save Model.
            current_model = spacy_model
        # Using the Default Model.
        else:
            current_model = self.default_model

        # See if the model is already saved.
        model_folder_path = join(self.spacy_folder, current_model)
        if isdir(model_folder_path):
            nlp = spacy.load(model_folder_path, disable=['ner'])
        # Otherwise, download and save the model.
        else:
            nlp = spacy.load(current_model, disable=['ner'])
            nlp.to_disk(model_folder_path)

        # Update the class model if we have to accept hyphenation.
        # https://spacy.io/usage/linguistic-features#how-tokenizer-works
        if hyphens:
            # Modify tokenizer infix patterns
            infixes = (
                    LIST_ELLIPSES
                    + LIST_ICONS
                    + [
                        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                        ),
                        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                        # Commented out regex that splits on hyphens between letters:
                        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
                    ]
            )
            # Update the Infix Patter for the class method.
            infix_re = compile_infix_regex(infixes)
            nlp.tokenizer.infix_finditer = infix_re.finditer

        # Make the model a class attribute.
        self.nlp = nlp

    def vocab_tokenizer(self, doc: str):
        """
        Tokenize a document creating a set with the lemmas of the words in the
        documents. This will be the vocabulary tokens of the document.

        Args:
            doc: String with the document from which we are going to create the
                vocabulary.
        Returns:
            List[str] containing the vocabulary tokens of the document.
        """
        # Check in case the document is empty.
        if not doc:
            return []

        # Tokenize the document.
        doc_tokens = self.nlp(doc)
        # Create vocabulary.
        doc_vocab = [
            token_word(token) for token in doc_tokens if vocab_acceptable(token)
        ]
        # The words in the vocabulary of the document.
        return doc_vocab

    def docs_vocab_tokenizer(self, docs: iter):
        """
        Tokenize an iterable of documents creating the vocabulary for each of
        the documents and returning them as a List of List[str], with each list
        containing the vocabulary tokens of each document.

        We use nlp.pipe(texts) to speed up the process of tokenization.

        Args:
            docs: Iterable with the documents we have to tokenize.
        Returns:
            List of List[str] with the vocabulary tokens of the documents.
        """
        # Create the default list of sets.
        docs_vocab_list = []

        # Use nlp.pipe to tokenize the documents.
        for doc_tokens in self.nlp.pipe(docs):
            doc_vocab = [
                token_word(token) for token in doc_tokens
                if vocab_acceptable(token)
            ]
            # Add the vocabulary of the document to the list.
            docs_vocab_list.append(doc_vocab)

        # The Vocabularies of the documents.
        return docs_vocab_list


def token_word(token: Token):
    """
    Given 'token' get the string representation most useful for the purpose of
    the tokenization, usually the word's lemma.

    Args:
        token: Token from where we are going to get the word representation.
    Returns:
        String with the corresponding text representation of the 'token'.
    """
    # Check if the word is Covid-19 or Covid-19 related.
    if token.lower_ == 'covid19':
        final_word = 'covid-19'
    elif 'covid19' in token.lower_ and 'covid19' != token.lower_:
        final_word = 'covid-19-related'
    elif 'covid-19' in token.lower_ and 'covid-19' != token.lower_:
        final_word = 'covid-19-related'
    # Check if we have a word in CAPITAL letters.
    elif token.is_upper:
        # We have an Acronym.
        final_word = token.text
    else:
        # Get the Lemma of the word.
        final_word = token.lemma_.lower().strip()

    # The appropriate representation for the word.
    return final_word


def vocab_acceptable(token: Token):
    """
    Check if a given token is a word, has 3 or more character, or is
    alphanumeric starting with a character.

    Args:
        token: Token that we are going analyze to see if it's appropriate for
            the vocabulary of the document.
    Returns:
        Bool indicating if the Token can be part of the document's vocabulary or
            not.
    """
    # No - if token length is too short or too long.
    if len(token.text) < 3 or len(token.text) > 25:
        return False
    # No - if the token represents a number.
    if token.like_num:
        return False
    # No - if the token is a Stop-word.
    if token.is_stop:
        return False
    # No - if the word doesn't have any vowels and is not an Acronym.
    vowels = {'a', 'e', 'i', 'o', 'u'}
    if not token.is_upper and not any(char in vowels for char in token.lower_):
        return False
    # Ok - if the word is all alphabetic.
    if token.is_alpha:
        return True
    # Ok - if token has covid-19 in it.
    if 'covid19' in token.lower_ or 'covid-19' in token.lower_:
        return True
    # Ok - if it has hyphens (-) and only alphabetic words.
    hyphenated_words = token.lower_.split('-')
    if all(map(lambda x: x.isalpha() or x in {'covid', '19', 'covid19'}, hyphenated_words)):
        return True
    # No by default - if none of the above cases apply.
    return False


if __name__ == '__main__':
    # Record Program Runtime.
    stopwatch = TimeKeeper()
    # Terminal Arguments.
    args = sys.argv

    # Create Tokenizer.
    print("\nCreating Tokenizer...")
    the_tokenizer = DocTokenizer(hyphens=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # # Test Tokenizing a Document.
    # the_doc = (
    #   'fun'
    # )
    # print("\nCurrent Document:")
    # pprint(the_doc, width=70)
    # the_vocab = the_tokenizer.vocab_tokenizer(the_doc)
    # print("\nDocument Tokens:")
    # pprint(the_vocab, width=70, compact=True)

    # Create a Random sample of Documents.
    the_num_docs = 10
    print(f"\nCreating a sample of {the_num_docs} documents...")
    the_sample = SampleManager(sample_size=the_num_docs, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Take document to tokenize.
    for the_doc in the_sample.corpus_title_abstracts():
        print("\nCurrent Document:")
        pprint(the_doc, width=70)

        # Ask to tokenize the document.
        the_input = input(
            "\nWould you like to tokenize this document?\n(q/quit to exit, n/next to skip): "
        )
        the_input = the_input.lower().strip()
        if the_input in {'q', 'quit', 'exit'}:
            break
        if the_input in {'n', 'next'}:
            continue

        # Tokenize and Display the Doc's tokens.
        the_vocab = the_tokenizer.vocab_tokenizer(the_doc)
        print("\nDocument Tokens:")
        pprint(the_vocab, width=70, compact=True)

        # Ask to continue.
        the_input = input("\nContinue? (q/quit to exit) ")
        if the_input in {'q', 'quit', 'exit'}:
            break

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
