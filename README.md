# -comparing2classifiers
Linear regression and logistic regression with Cross validation, t test implemented
On the Susy data set, the steps I took to perform CV are as follows:
1.	Divided my data in 10 fold with out repeating data
2.	Hold out 1 fold as test set and others as trainset.
3.	Train and record the test set result.
4.	Perform 2 and 3 on each fold in turn as test set
5.	Calculate the average and standard deviation of all the fold results

I did the above steps on 30 folds of repetitive data to ensure exhaustible algorithm. 
I get L1 an L2 with list of accuracies for 30 folds in linear learner and Logistic learner;
I pass that on to scipy.stats.ttest_ind which gives me a tail of P values. If we observe a large p-values ie., larger than 0.05  then we cannot reject the null hypothesis of identical average scores. If the p-value is smaller than the threshold, e.g. 5%, then we reject the null hypothesis of equal averages. 
If once rejected, I check for the mean of accuracies in both the lists L1 and L2, whichever is better, I return that algorithm as better performing one.
I ran my algorithm a couple of times over Linear and Logistic regression. My hyperparameter I am trying to resolve with cross validation is Lamda. I am running it for some 10 values of lamda “0.1,0.01,0.05,.005,.0005,.001,.0001,.00001,.000001,1”. 
With best value from each fold, I am running it on the whole set to predict the test set. I get L1 and L2 and for me when I pass it to t test, Null hypothesis fails because of low p values. Also I get my Logistic regression to be performing better mostly. 
P values for one of the run ;  -7.5894,0.016921
Which are lesser than .05, hence falsifying the null hypothesis. 
Mean was checked which returned Logistic as to be a better classifier.

