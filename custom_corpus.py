# Gelin Eguinosa Rosique
# 2022

from os import listdir
from os.path import isdir, isfile, join

from topic_corpus import TopicCorpus

# Testing Imports.
import sys
from pprint import pprint
from time_keeper import TimeKeeper


class CustomCorpus(TopicCorpus):
    """
    Class to load a custom corpus from .txt files inside a folder.
    """

    def __init__(self, dir_path: str):
        """
        Load the documents inside the folder 'dir_path' and save their titles
        and abstracts.
        """
        # Check we have valid folder.
        if not isdir(dir_path):
            raise NameError(f"The path <{dir_path}> is not a folder.")

        # Load the content of the available txt files.
        docs_content = {}
        for item_name in listdir(dir_path):
            # Check if it ends with txt.
            if not item_name.endswith('.txt'):
                continue
            # Skip if it's not a file.
            item_path = join(dir_path, item_name)
            if not isfile(item_path):
                continue
            # It's a txt file - Save Content of document.
            with open(item_path, 'r') as f:
                doc_content = f.read()
            doc_id = item_name[:-4]
            docs_content[doc_id] = doc_content

        # Separate Title & Abstract of the documents.
        corpus_index = {}
        for doc_id, doc_content in docs_content.items():
            # Separate Doc Content by Lines.
            doc_lines = doc_content.split('\n')

            # First Not Empty Line is the Title.
            # The body start at the Second Not Empty Line.
            doc_title = ''
            body_lines = []
            for doc_line in doc_lines:
                if not doc_title and not doc_line:
                    continue
                elif not doc_title and doc_line:
                    doc_title = doc_line
                elif not body_lines and not doc_line:
                    continue
                else:
                    body_lines.append(doc_line)
            # Join the Lines of the Body Text to form Abstract.
            doc_abstract = '\n'.join(body_lines)

            # Save the Document if it has both title & abstract.
            if not doc_title or not doc_abstract:
                continue
            corpus_index[doc_id] = {
                'title': doc_title,
                'abstract': doc_abstract
            }

        # Save the Corpus Index.
        self.corpus_index = corpus_index

    def corpus_identifier(self):
        """
        Currently this corpus can't be saved or loaded, so return an empty
        identifier.
        """
        return ''

    @property
    def doc_ids(self):
        """
        List of the IDs of the Documents in the Corpus.
        """
        doc_ids = list(self.corpus_index.keys())
        return doc_ids

    def doc_title(self, doc_id: str):
        """
        Title of the Document 'doc_id'.
        """
        doc_index = self.corpus_index[doc_id]
        doc_title = doc_index['title']
        return doc_title

    def doc_abstract(self, doc_id: str):
        """
        Abstract of the Document 'doc_id'.
        """
        doc_index = self.corpus_index[doc_id]
        doc_abstract = doc_index['abstract']
        return doc_abstract

    def doc_body_text(self, doc_id: str):
        """
        Body Text is not supported in this type of Corpus. Return empty string.
        """
        return ''

    def doc_specter_embed(self, doc_id: str):
        """
        Specter Embedding is not supported in this type of Corpus. Return empty
        list.
        """
        return []

    def load_embeddings_dict(self):
        """
        Embeddings Dict Not supported.
        """
        pass

    def unload_embeddings_dict(self):
        """
        Embeddings Dict Not supported.
        """
        pass


if __name__ == '__main__':
    # Record Runtime of the Program.
    _stopwatch = TimeKeeper()
    # Terminal Arguments.
    _args = sys.argv

    # Create Custom Corpus.
    _custom_corpus = CustomCorpus('temp_data')
    for _doc_id in _custom_corpus.doc_ids:
        print(f"\nDoc <{_doc_id}>:")
        # --------------------------------------------------
        # print("Title:", _custom_corpus.doc_title(_doc_id))
        # print("Abstract:")
        # print(_custom_corpus.doc_abstract(_doc_id))
        # --------------------------------------------------
        pprint(_custom_corpus.doc_title_abstract(_doc_id))

    print("\nDone.")
    print(f"[{_stopwatch.formatted_runtime()}]\n")
