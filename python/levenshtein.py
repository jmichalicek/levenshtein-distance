"""
Find the minimum edit distance between two words using the Levenshtein Distance in Python
"""
from __future__ import print_function, unicode_literals, division, absolute_import

# there are better options, but this is built in
import argparse

# These values are currently ignored
REPLACE_COST = 2
INSERT_COST = 1
REMOVE_COST = 1


class Implementation(object):
    FULL = 'full'
    SHORT_WORD = 'short-word'


def init_matrix(word_one, word_two):
    """
    Initialize the base matrix of the words before calculating
    the cost for each change.
    """
    if len(word_one) == 0 or len(word_two) == 0:
        return 0
    
    # init the first row using word_one
    distance_matrix = [[0] + [idx for idx, _ in enumerate(word_one, start=1)]]
    word_one_len = len(word_one)
    word_two_len = len(word_two)

    # init the rest of the rows using word_two
    # giving a matrix which is word_one_len columnx and word_two_len rows
    for idx, character in enumerate(word_two, start=1):
        prefix = [idx for x in range(idx)]
        row = prefix + distance_matrix[0][idx:]
        distance_matrix.append(row)

    return distance_matrix


def full_matrix(word_one, word_two, print_matrix=False):
    """
    Less efficient implementation which iterates over everything.
    You can easily see each step and it makes a nice printable matrix
    where all values make sense.

    This version first creates a base matrix, so iterates over each word once.
    Then iterates over each row in the matrix and then each column in that row
    comparing each character and updating the matrix.

    Literally implements the basic algorithm
    * initialize first row with word
    * initialize first column with word_two
    * for i in 0..word_one_len
        for j in 0..word_two_len
          if word_one[i] == word_two[j]
            action_cost = 0 
          else
            action_cost = 1
        matrix[i][j] = min(matrix[j-1], matrix[i-1], matrix[i-1][j-1]) + action_cost
      return matrix[i][j]
    """
    distance = 0
    word_one_len = len(word_one)
    word_two_len = len(word_two)

    if word_one == word_two:
        return distance
    elif not word_one or not word_two:
        return max(word_one_len, word_two_len)

    distance_matrix = init_matrix(word_one, word_two)
    # sinc this only looks at things previously created, it seems like
    # I should be able to just do this in the previous iteration
    # over `word_two` and combine initializing the matrix with this iteration.
    # I think with minimal special casing this could just become O(N) as well
    # by just iterating the length of the shorter of the two strings
    # and then tacking on however many inserts are needed to make the longer string.
    # That would kill printing the final matrix, of course.
    for idx, row in enumerate(distance_matrix):
        # switch this of an n^2 iteration over the row now
        if idx == 0:
            continue
        # row is length of word_one + 1
        for idx2, col in enumerate(row):
            # continue if idx2 == 0?
            action_cost = 1
            # if the letters we are comparing are the same, then there is no cost to this action
            if idx2 <= word_one_len and idx <= word_two_len and word_one[idx2-1] == word_two[idx-1]:
                action_cost = 0

            before = row[idx2-1]
            above = distance_matrix[idx-1][idx2]
            angle = distance_matrix[idx-1][idx2-1]

            # add the action cost to lowest of previous cost changes
            current_cost = min(before, above, angle) + action_cost 
            row[idx2] = current_cost

    if print_matrix:
        print([' ', ' '] + list(word_one))

        for idx, x in enumerate(distance_matrix):
            if idx > 0:
                line = [word_two[idx-1]]    
            else:
                line = [' ']
            line += [str(y) for y in x]
            print(line)

    return current_cost


def shortest_word(word_one, word_two, *args, **kwargs):
    """
    Slightly optimized implementation

    This version should be somewhat more performant but may not be as clear
    and does not make as obvious of a matrix to print.

    In this the shorter word  is iterated over to make the first row of the matrix,
    then each additional row is built while iterating over the shorter word, ensuring
    a square matrix of the shortest word length.
    It also only compares the values at the specific indexes instead of each character
    in each word instead of each character.  Any additional characters in the longer word
    we know will be inserts, so the insert cost for them is just added on.

    If support for printing the matrix is dropped then this could be further optimized to
    only keep two rows in memory.
    """
    distance = 0
    word_one_len = len(word_one)
    word_two_len = len(word_two)
    shorter = word_one if word_one_len < word_two_len else word_two
    longer = word_one if word_one_len >= word_two_len else word_two
    shortest_len = len(shorter)

    if word_one == word_two:
        return distance
    elif not word_one or not word_two:
        return max(word_one_len, word_two_len)

    distance_matrix = [[0] + [idx for idx, _ in enumerate(shorter, start=1)]]
    for idx, row in enumerate(shorter, start=1):
        action_cost = 1

        prefix = [idx for x in range(idx)]
        row = prefix + distance_matrix[0][idx:]
        if idx <= shortest_len and word_one[idx-1] == word_two[idx-1]:
            action_cost = 0

        before = row[idx-1]
        above = distance_matrix[idx-1][idx]
        angle = distance_matrix[idx-1][idx-1]

        distance = min(before, above, angle) + action_cost
        row[idx] = distance
        distance_matrix.append(row)

    distance += abs(word_one_len - word_two_len)
    return distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # should be able to use metavar=('word1', 'word2') but current python I am using
    # does not seem to like it
    parser.add_argument('words', nargs=2,
                        help='The two words to compare')
    parser.add_argument('-i', '--implementation', dest='implementation',
                        default=Implementation.FULL,
                        choices=[Implementation.FULL, Implementation.SHORT_WORD],
                        help='The implementation of levenshtein to use')

    args = parser.parse_args()
    words = args.words
    if args.implementation == Implementation.FULL:
        lev_func = full_matrix
    elif args.implementation == Implementation.SHORT_WORD:
        lev_func = shortest_word

    print(lev_func(*words))
