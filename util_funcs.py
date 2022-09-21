# Gelin Eguinosa Rosique
# 2022

import numpy as np
from numpy.linalg import norm
from bisect import bisect

from extra_funcs import progress_bar

# Test imports.
import sys
import random
from pprint import pprint
from time_keeper import TimeKeeper


def dict_ndarray2list(embeds_dict: dict, show_progress=False):
    """
    Transform the values of the dictionary 'embeds_dict' from Numpy.ndarray to
    list.

    Args:
        embeds_dict: Dictionary(ID -> List[Numpy.ndarray]) with the embeddings
            we want to transform to list.
        show_progress: Bool representing whether we show the progress of
            the method or not.
    Returns:
        Dictionary(ID -> List[Float]) with the embeddings of the dictionary
            with type list instead of Numpy.ndarray.
    """
    # Progress Variables.
    count = 0
    total = len(embeds_dict)
    # Default Dict.
    new_embeds_dict = {}
    for item_id, embed in embeds_dict:
        new_embeds_dict[item_id] = embed.tolist()
        if show_progress:
            count += 1
            progress_bar(count, total)
    # The Transformed Dictionary.
    return new_embeds_dict


def dict_list2ndarray(embeds_dict: dict, show_progress=False):
    """
    Transform the values of the dictionary 'embeds_dict' from List to
    Numpy.ndarray.

    Args:
        embeds_dict: Dictionary(ID -> List[Float]) with the embeddings we want
            to transform to Numpy.ndarray.
        show_progress: Bool representing whether we show the progress of
            the method or not.
    Returns:
        Dictionary(ID -> List[Numpy.ndarray]) with the embeddings of the
            dictionary with type Numpy.ndarray instead of List.
    """
    # Progress Variables.
    count = 0
    total = len(embeds_dict)
    # Default Dict.
    new_embeds_dict = {}
    for item_id, embed in embeds_dict:
        new_embeds_dict[item_id] = np.array(embed)
        if show_progress:
            count += 1
            progress_bar(count, total)
    # Transformed Dictionary.
    return new_embeds_dict


def cos_sim(a: np.ndarray, b: np.ndarray):
    """
    Calculate the cosine similarity between the vectors 'a' and 'b'.

    Args:
        a: Numpy.ndarray containing one of the vectors embeddings.
        b: Numpy.ndarray containing one of the vectors embeddings.

    Returns:
        Float with the cosine similarity between the two vectors.
    """
    # Use Numpy.
    result = np.dot(a, b) / (norm(a) * norm(b))
    # Transform from float32 to float (float32 is not JSON serializable)
    result = float(result)
    return result


def closest_vector(embedding, vectors_dict: dict):
    """
    Given the embedding of a document or word and a dictionary containing a
    group of IDs with their embeddings. Find the closest vector to the given
    embedding using cosine similarity.

    Args:
        embedding: Numpy.ndarray with the embedding of the word or document that
            we are using to find the closest vector.
        vectors_dict: Dictionary containing the vectors with their IDs as
            keys and their embeddings as values.

    Returns:
        A tuple with the ID of the closest vector and its similarity to the
            'embedding' we received as parameter.
    """
    # Use iter to get the vectors IDs and their embeddings.
    vector_iter = iter(vectors_dict.items())

    # Get cosine similarity to the first vector.
    closest_vector_id, vector_embed = next(vector_iter)
    max_similarity = cos_sim(embedding, vector_embed)

    # Iterate through the rest of the vectors.
    for vector_id, vector_embed in vector_iter:
        new_similarity = cos_sim(embedding, vector_embed)
        if new_similarity > max_similarity:
            # New Closer Vector
            closest_vector_id = vector_id
            max_similarity = new_similarity

    # The closest vector ID with its similarity to the 'embedding'.
    return closest_vector_id, max_similarity


def top_n(id_values: iter, n=50, top_max=True, show_progress=False):
    """
    Given a list of tuples (IDs, Values) find the top 'n' tuples in the list
    using their values.

    Args:
        id_values: List[Tuple(IDs, Values)] with the IDs and Values we want to
            use to get the top 'n' values.
        n: Int with the amount of Top IDs we are going to return.
        top_max: Bool indicating the Maximum value is the Top value when 'True',
            and the Minimum value is the Top value when 'False'.
        show_progress: Bool representing whether we show the progress of
            the method or not.
    Returns:
        List[Tuple(IDs, Values)] with the top 'n' tuples in the list.
    """
    # Top Values list.
    top_ids_values = []
    sort_values = []

    # Iterate though the elements of the dictionary to get the top n.
    count = 0
    total = len(id_values)
    for item_id, value in id_values:
        # Check if we want the top maximum values, if that's the case multiply
        # by -1 so the max values are now the minimum values.
        if top_max:
            sort_value = -1 * value
        else:
            sort_value = value
        # Fill the list with values first.
        if len(top_ids_values) < n:
            index = bisect(sort_values, sort_value)
            top_ids_values.insert(index, (item_id, value))
            sort_values.insert(index, sort_value)
        # If the list is full, check if check is this value is better.
        elif len(top_ids_values) >= n and sort_value < sort_values[-1]:
            # Delete last element of the lists.
            del top_ids_values[-1]
            del sort_values[-1]
            # Add the current value to the sorted lists.
            index = bisect(sort_values, sort_value)
            top_ids_values.insert(index, (item_id, value))
            sort_values.insert(index, sort_value)
        # Progress of the search.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # The Top tuples of (IDs, Values).
    return top_ids_values


if __name__ == '__main__':
    # Record Program Runtime.
    stopwatch = TimeKeeper()
    # Terminal Arguments.
    args = sys.argv

    # Create a list of tuples to get the Top N elements.
    my_origin = random.sample(range(12, 100), 20)
    my_tuples = [(n, n) for n in my_origin]
    print("\nElements to Sort:")
    pprint(my_tuples)

    my_top_num = 10
    print(f"\nTop {my_top_num} elements:")
    my_tops = top_n(my_tuples, n=my_top_num, top_max=True)
    pprint(my_tops)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
