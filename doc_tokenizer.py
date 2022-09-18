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
        documents. This would be the vocabulary of the document.

        Args:
            doc: String with the document from which we are going to create the
                vocabulary.

        Returns:
            Set(str) containing the words in the vocabulary of the document.
        """
        # Check in case the document is empty.
        if not doc:
            return set()

        # Tokenize the document.
        doc_tokens = self.nlp(doc)
        # Create vocabulary.
        doc_vocab = {
            token_word(token) for token in doc_tokens if is_acceptable(token)
        }
        # The words in the vocabulary of the document.
        return doc_vocab


def token_word(token: Token):
    """
    Get the best string representation for 'token' in the document.
    """
    # return token.lemma_.lower()
    pass


def is_acceptable(token: Token):
    """
    Check if a given token is acceptable as a word of the vocabulary in the
    corpus.
    """
    pass


if __name__ == '__main__':
    # Record Program Runtime.
    stopwatch = TimeKeeper()
    # Terminal Arguments.
    args = sys.argv

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
