# subreddit-classifier
Machine Learning Agent - Subreddit Classifier. 

The main objective of this project is to build a machine learning agent that can predict from which subreddit an unlabelled post comes from.
The classification can be performed on any number of subreddits for any number of records. 
For the given project the subreddits chosen are Computer Science, Data Science and GRE. For each given subreddit 500 latest posts are fetched.
The task of collecting the posts is done using the PRAW API.
The data is split into three different parts - 50%, 25%, 25% as training, development and testing.
Used scikit-learn and nltk tools to transform the text and subreddit fields of each post into scikit-learn feature vectors and labels.
Feature extraction such as stemming, lemming, removal of stop words performed using nltk library.
Created a bag of words using TF-ID vectorizaiton.
Implemented the Random Forest and Support Vector Machine.

For Data Visualization, wordclouds of each subreddit data are created.

In order to test the performance, different performance paramters such as ROC Curve, Confusion Matrix, Accuracy Score, Classification Report and Precision & Recall models are implemented.

Observation: Classification is more precise, when the subreddits are unrelated to each other.(i.e they are more different). Number of records also affect the performance. The accuracy is low when the subreddits are closely related to each other and less number of records are taken into consideration. The more data you download, the better your performance will be.

Programming Language Used : Python

Requirements :
	To fetch the new data from reddit an active Internet connection is required.

Installations :
	1. Install PRAW - a Reddit API for Python:
	https://github.com/praw-dev/praw
	
	2. To perform removal of stop words install the set of stopwords using nltk:
	import nltk
	nltk.download("stopwords")
	
System Requirements :
	Python Installed, preffered IDE - PyCharm.

For Execution :
	How to run :
		1.	Open terminal.
		2.	Go to the directory of program.
		3.	Run as follows :
				python 'PROGRAM_NAME'.py
				eg : python random_forest.py
				     python svm.py
				     pthon subreddits_wordcloud.py
				     python subreddit_roc_curve.py					
