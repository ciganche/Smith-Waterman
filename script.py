# Algorithms for Bioinformatics 2021
# Author: Aleksa Krsmanovic
# email: aleksa.krsmanovic@studenti.unitn.it
# date: 27.7.2021

import sys
import argparse
import re
import numpy as np
import collections
import pandas as pd
from argparse import RawTextHelpFormatter

GAP_STRING = "_"


#  -----****-----
#  -----FUNS-----
#  -----****-----
def read_params(args):
    parser = argparse.ArgumentParser(description="""Smith-Waterman algorithm for optimal local sequence alignment. Use fasta files as input or type the sequences manually.
        
    Example 1: python script.py -r TACGGGCCCGCGG -q TAGCCCTATCGGTCG -m 5 -mm -2 -g -2 -ai test_info.tsv -as alignment_strings.txt
    Example 2: python script.py -r reference_demo.fasta -q TAGCCCTATCGGTCG""", formatter_class=RawTextHelpFormatter)
    arg = parser.add_argument
    parser._optionals.title = "Arguments"
    arg("--reference", "-r", required=True, type=str,
        help="[REQUIRED] The reference sequence or a path to the fasta file containing it."
             " If using a file, The only extension allowed is '.fasta'. The fasta file must contain only one sequence.")
    arg("--query", "-q", required=True, type=str,
        help="[REQUIRED] The query sequence or a path to the fasta file containing it."
             " If using a file, The only extension allowed is '.fasta'. The fasta file must contain only one sequence.")
    arg("--match", "-m", default=1, type=float,
        help="Scoring value of a match. Default is 1.")
    arg("--mismatch", "-mm", default=-1, type=float,
        help="Scoring value of a mismatch. Default is 1.")
    arg("--gap", "-g", default=-1, type=float,
        help="Scoring value of a gap. Default is -1.")
    arg("--aligment_info", "-ai", default="aligned_sequence_info.tsv", type=str,
        help="Path and name of the tsv file containing information about the alignment. Default is script_directory/aligned_sequence_info.tsv.")
    arg("--alignment_strings", "-as", default="alignment_strings.txt", type=str,
        help="Path and name of the text file containing alignment strings. Default is script_directory/alignment_strings.txt.")
    return vars(parser.parse_args())


# sequences can be provided as a fasta filer or as simple text.
# If the argument ends with .fasta, it is assumed a file is used as input.
def parse_sequences(sequence_arg):
    if sequence_arg.endswith(".fasta"):    # load form file
        f = open(sequence_arg, "r")
        f.readline()    # skip >ID line
        sequence = f.readline().strip()
        f.close()
    else:   # load from command line
        sequence = sequence_arg.strip()

    # raise error if sequence contains anything but nucleotides
    match_obj = re.search("[^ATCGatcg]", sequence)
    if match_obj is not None:
        raise Exception("Provided sequence contains invalid characters: " + str(match_obj.group()))

    if len(sequence) < 3:
        raise Exception("The input sequence need to be at least 3 nucleotides long.")

    sequence = sequence.upper()
    return sequence


def init_and_fill(query, reference, match, mismatch, gap):
    # init matrix, +1 to the lenght of the sequences is for the gap
    rows = len(query) + 1
    cols = len(reference) + 1
    # score_matrix is keeping the scores
    score_matrix = np.zeros((rows, cols))

    # visited_matrix keeps track of all score matrix values used to create an alignment
    # these values should not be used again as a starting point for a new alignment
    visited_matrix = [[False for col in range(cols)] for row in range(rows)]

    # path_matrix is a matrix explaining from which value is the CURRENT value derived. All NON-ZERO values are tracked.
    # Value 1 is LEFT, 2 is DIAGONAL, 3 is UP.
    path_matrix = [[[] for col in range(cols)] for row in range(rows)]

    # matrix filling, ommit first row and column
    for i in range(1, rows):
        for j in range(1, cols):
            match_or_miss = match if query[i - 1] == reference[j - 1] else mismatch
            up_left_score = score_matrix[i - 1][j - 1] + match_or_miss
            up_score = score_matrix[i - 1][j] + gap
            left_score = score_matrix[i][j - 1] + gap

            score_matrix[i][j] = max(0, left_score, up_left_score, up_score)

            # draw paths for non-zero values. If multiple values are qeual, traceback to multiple values is possible. All posible transitions are saved.
            if score_matrix[i][j] != 0:
                if score_matrix[i][j] == left_score:
                    path_matrix[i][j].append(1)  # LEFT
                if score_matrix[i][j] == up_left_score:
                    path_matrix[i][j].append(2)  # DIAGONAL
                if score_matrix[i][j] == up_score:
                    path_matrix[i][j].append(3)  # UP

    return score_matrix, visited_matrix, path_matrix


def get_max_score(score_matrix):
    index = np.unravel_index(np.argmax(score_matrix, axis=None), score_matrix.shape)
    value = np.max(score_matrix)
    return index, value


def get_sorted_scores(score_matrix, match):
    dict = {}

    # transform matrix into a dictionary key= score_matrix indicies : value=score_matrix value
    # all values greater than a single match are taken
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            if score_matrix[i][j] > match:
                dict[(i, j)] = score_matrix[i][j]
    # sort dictionary by value in descending order and return a sorted dictionary
    sorted_x = reversed(sorted(dict.items(), key=lambda kv: kv[1]))
    return collections.OrderedDict(sorted_x)

# finding all possible alignments using a SINGLE starting point is analogous to finding all paths between two nodes in an directed, acyclic graph.
# find all possible paths from STARTING_POINT to 0 using depth-first-search traversal
# while finding the paths alignments are created.
def traceback(starting_point, start_point_score, score_matrix, visited_matrix, path_matrix, sequence1, sequence2, aligned_sequence_list):
    aligned_sequence1 = ""
    aligned_sequence2 = ""
    # in DFS, for all possible transitions recursion is called, then after the recursion is finished the transition is undone.
    traceback_single_step(starting_point, start_point_score, score_matrix, visited_matrix, path_matrix, aligned_sequence_list, aligned_sequence1, aligned_sequence2, sequence1, sequence2)


def traceback_single_step(position, start_point_score, score_matrix, visited_matrix, path_matrix, aligned_sequence_list, aligned_sequence1, aligned_sequence2, sequence1, sequence2):
    i, j = position[0], position[1]
    # recursion exit condition - reaching 0 in the scoring matrix
    if score_matrix[i][j] == 0:
        aligned_sequence_list.append(get_alignment_info(start_point_score, aligned_sequence1, aligned_sequence2))
    else:
        visited_matrix[i][j] = True

        # for all possible transitions recursion is called
        for direction in path_matrix[i][j]:

            if direction == 2:  # DIAGONAL movement - both aligned
                aligned_sequence1 = sequence1[i-1] + aligned_sequence1
                aligned_sequence2 = sequence2[j-1] + aligned_sequence2
                i = i - 1
                j = j - 1
            elif direction == 3:    # UP movement - gap in sequence 2
                aligned_sequence1 = sequence1[i-1] + aligned_sequence1
                aligned_sequence2 = GAP_STRING + aligned_sequence2
                i = i - 1
            elif direction == 1:    # LEFT movement - gap in sequence 1
                aligned_sequence1 = GAP_STRING + aligned_sequence1
                aligned_sequence2 = sequence2[j-1] + aligned_sequence2
                j = j - 1
            else:
                raise Exception("Error detected in path matrix values. Wrongly constructed.")

            # call recursion on next position (i, j)
            traceback_single_step((i, j), start_point_score, score_matrix, visited_matrix, path_matrix, aligned_sequence_list, aligned_sequence1, aligned_sequence2, sequence1, sequence2)

            # UNDO - revert to original position after calling recursion: 1) remove added nucleotied/gap. 2) reposition the indicies in the scoring matrix
            aligned_sequence1 = aligned_sequence1[1:]
            aligned_sequence2 = aligned_sequence2[1:]
            if direction == 2:  # undo DIAGONAL movement
                i = i + 1
                j = j + 1
            elif direction == 3:  # undo UP movement
                i = i + 1
            elif direction == 1:  # undo LEFT movement
                j = j + 1


def get_alignment_info(start_point_score, aligned_sequence1, aligned_sequence2):

    if len(aligned_sequence1) != len(aligned_sequence2):
        raise Exception("Error while aligning. Alignments not of same length.")

    ret_val = [start_point_score, aligned_sequence1, aligned_sequence2]
    insertions = aligned_sequence2.count(GAP_STRING)
    deletions = aligned_sequence1.count(GAP_STRING)

    matches = 0
    mismatches = 0
    for i in range(len(aligned_sequence1)):
        if aligned_sequence1[i] == GAP_STRING or aligned_sequence2[i] == GAP_STRING:
            continue
        if aligned_sequence1[i] == aligned_sequence2[i]:
            matches = matches + 1
        else:
            mismatches = mismatches + 1

    alignment_string = get_alignment_string(aligned_sequence1, aligned_sequence2)

    ret_val.append(alignment_string)
    ret_val.append(matches)
    ret_val.append(mismatches)
    ret_val.append(insertions)
    ret_val.append(deletions)
    ret_val.append(len(aligned_sequence1))

    return ret_val


def get_alignment_string(aligned_sequence1, aligned_sequence2):

    middle_line = ""
    for j in range(len(aligned_sequence1)):
        if aligned_sequence1[j] == GAP_STRING or aligned_sequence2[j] == GAP_STRING:
            middle_line = middle_line + " "
        elif aligned_sequence1[j] == aligned_sequence2[j]:
            middle_line = middle_line + "*"
        else:
            middle_line = middle_line + "|"

    return aligned_sequence1 + "\n" + middle_line + "\n" + aligned_sequence2


def print_alignment_strings(aligned_sequence_df, selected_alignments_df, alignment_strings_filename):

    # print only the selected on stdout
    for index, row in selected_alignments_df.iterrows():
        item = ""
        item = "\n" + item + str(index) + "   Score: " + str(row["Score"]) + ":\n" + row["Alignment_str"] + "\n"
        print(item)

    # print all string in a file
    content = ""
    for index, row in aligned_sequence_df.iterrows():
        item = ""
        item = "\n" + item + str(index) + "   Score: " + str(row["Score"]) + ":\n" + row["Alignment_str"] + "\n"

        content = content + item

    f = open(alignment_strings_filename, "w")
    f.write(content)
    f.close()


#  -----****-----
#  -----MAIN-----
#  -----****-----
if __name__ == "__main__":
    par = read_params(sys.argv)
    reference = parse_sequences(par["reference"])
    query = parse_sequences(par["query"])
    match = par["match"]
    mismatch = par["mismatch"]
    gap = par["gap"]
    alignment_info_filename = par["aligment_info"]
    alignment_strings_filename = par["alignment_strings"]


    # path_matrix denotes which transitions are allowed
    # visited_matrix is used to ensure we are not choosing staring points already covered by another alignment
    score_matrix, visited_matrix, path_matrix = init_and_fill(query, reference, match, mismatch, gap)

    # create a sorted dictionary of key=sore_matrix indicies : value=score_matrix value
    sorted_dict = get_sorted_scores(score_matrix, match)

    aligned_sequence_list = [] # a list of alignment information to be turned into a pandas dataframe
    for start_point, start_point_score in sorted_dict.items():
        i, j = start_point[0], start_point[1]
        if visited_matrix[i][j]:
            continue
        traceback(start_point, start_point_score, score_matrix, visited_matrix, path_matrix, query, reference, aligned_sequence_list)

    aligned_sequence_df = pd.DataFrame(aligned_sequence_list,
                                       columns=["Score", "Query_alignment", "Ref_alignment", "Alignment_str", "Matches",
                                                "Mismatches", "Insertions", "Deletions", "Length"])
    aligned_sequence_df.drop("Alignment_str", axis=1).to_csv(alignment_info_filename, sep="\t")

    # write and print output
    print("Match: " + str(match) + ", Mismatch: " + str(mismatch) + ", Gap: " + str(gap) + "\n")
    print("Best alignments:")  # PROMENI

    # print on stdout only the best alignments with the highest score
    max_score = max(aligned_sequence_df["Score"])
    selected_alignments_df = aligned_sequence_df[aligned_sequence_df["Score"] == max_score]

    # print alignment strings of selected alignments on stdout
    print_alignment_strings(aligned_sequence_df, selected_alignments_df, alignment_strings_filename)
    # print alignment info of selected alignments on stdout
    print(selected_alignments_df.drop("Alignment_str", axis=1))

    print("\nFull alignment info output saved in: " + alignment_info_filename)
    print("All alignment strings saved in: " + alignment_strings_filename)
