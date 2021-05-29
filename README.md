# venus

####Low Rank Matrix Completion

Matrix Completion is the task of filling the missing elements in a matrix. The
famous example is Netflix challenge: Given a rating matrix in which (i, j) element
refers to rating for ith movie by jth user. There are a lot of missing entries in the
matrix as a particular user is expected to rate only a few of the movies. So matrix
completion is used in such cases to fill out the missing entries. Netflix used matrix
completion to recommend new movies to the users. I used matrix completion on
Amazon product ratings data to predict the missing ratings by the users. The below two
algorithms are implemented for low-rank matrix completion:

1. Iterative Singular Value Thresholding
2. Alternative Minimization    