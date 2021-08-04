# Smith-Waterman local sequence alignment
An optimal algorithm for local sequence alignment with an UNIX-like user interface.\
\
Finds all possible alignments between two sequences by utilizing depth first search. This means overlapping alignments are returned but without any subalignments, like in the case of https://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Smith-Waterman.
All found alignments are saved to a tsv file, while the best ones are shown on stdout. 

Run ```python script.py -h``` to see all options. 
