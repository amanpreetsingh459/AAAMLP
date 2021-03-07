# Notes
1. Use kfold cross validation with random split
2. Do not use kfold with skewed(unbalanced data), use stratified kfold instead
3. stratified kfold cannot be used for regression problems directly
4. To use stratified kfold for regression problems, convert the dataset into classification one. One such way to do that is - divide the dataset into bins based on target values, then use those bins as classification targets
5. decide the number of bins based on the Sturge’s Rule:-
 >Number of Bins = 1 + log2(N)
6. Golden Rule: Use "stratified kfold" when things seem blind
