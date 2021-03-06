CS 760 Homework Assignment

a program that implements both naive Bayes and TAN (tree-augmented naive Bayes). The program is intended for binary 
classification problems. All of the variables are discrete valued. The program can handle an arbitrary number of variables with 
possibly different numbers of values for each variable. Laplace estimates (pseudocounts of 1) are used when estimating all 
probabilities.

For the TAN algorithm:

Prim's algorithm is used to find a maximal spanning tree. The first variable in the input file is used to initialize the process.
If there are ties in selecting maximum weight edges, the following preference criteria is used: 
(1) prefer edges emanating from variables listed earlier in the input file, (2) if there are multiple maximum weight edges
emanating from the first such variable, prefer edges going to variables listed earlier in the input file.
To root the maximal weight spanning tree, pick the first variable in the input file as the root.

The program should be called bayes and should accept four commandline arguments as follows:
bayes <train‐set‐file> <test‐set‐file> <n|t>
where the last argument is a single character (either 'n' or 't') that indicates whether to use naive Bayes or TAN.

The program determines the network structure (in the case of TAN) and estimate the model parameters using the given training set, 
and then classify the instances in the test set. 

Output:
The structure of the Bayes net by listing one line per variable in which (i) the name of the variable, (ii) the names of its 
parents in the Bayes net (for naive Bayes, this will simply be the 'class' variable for each other variable) separated by 
whitespace. One line for each instance in the testset (in the same order as this file) indicating (i) the predicted class,
(ii) the actual class, (iii) and the posterior probability of the predicted class (rounded to 12 digits after
the decimal point). The number of the testset examples that were correctly classified.
