import praw
import sklearn
import nltk
import pandas as pd
import re
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Defining a list of stopwords
my_stopwords = stopwords.words('english')

# Connecting to the reddit script app through PRAW API
reddit = praw.Reddit(client_id='Nuhb2V_ONfbRAg',\
                     client_secret='ao8oVLq-ZEz1a-9leqg1FqGdCIg',\
                     user_agent='redditApp1',\
                     username='ketand2017',\
                     password='passwordfis'
                     )

# The subreddits from which data is collected
subreddit1 = reddit.subreddit('computerscience')
subreddit2 = reddit.subreddit('datascience')
subreddit3 = reddit.subreddit('GRE')

# Collecting the latest 500 posts from each subreddit
new_subreddit1 =subreddit1.new(limit=500)
new_subreddit2 =subreddit2.new(limit=500)
new_subreddit3 =subreddit3.new(limit=500)

# Storing the subreddits data into json format
first_subreddit_data = {"body":[],"subreddit":[]}
second_subreddit_data = {"body":[],"subreddit":[]}
third_subreddit_data = {"body":[],"subreddit":[]}

# Appending the posts and label for first subreddit data
for post in new_subreddit1:
    first_subreddit_data["body"].append(post.selftext)
    first_subreddit_data["subreddit"].append(post.subreddit.display_name)

# Appending the posts and label for second subreddit data
for post in new_subreddit2:
    second_subreddit_data["body"].append(post.selftext)
    second_subreddit_data["subreddit"].append(post.subreddit.display_name)

# Appending the posts and label for third subreddit data
for post in new_subreddit3:
    third_subreddit_data["body"].append(post.selftext)
    third_subreddit_data["subreddit"].append(post.subreddit.display_name)

# Converting the collected data into dataframe
data1 = pd.DataFrame(first_subreddit_data)
data2 = pd.DataFrame(second_subreddit_data)
data3 = pd.DataFrame(third_subreddit_data)

# Storing the data for each subreddit into json file
data1.to_json("subredditdata1.json",orient='split')
data2.to_json("subredditdata2.json",orient='split')
data3.to_json("subredditdata3.json",orient='split')

# Merging all the data
data = data1.append(data2)
data = data.append(data3)

# Storing the data into json file
data.to_json("subredditdata.json",orient='split')
data['subreddit'].replace({'computerscience': 0, 'datascience': 1,'GRE':2}, inplace=True)

#Shuffling the data
data = data.sample(frac=1)

#Splitting the data into 50, 25, 25
print('Splitting the data in training 50%, development 25%, testing 25%')
training_data = data.iloc[:(len(data)//2)-1,:]
development_data = data.iloc[(len(data)//2)-1:int(0.75*len(data)),:]
testing_data = data.iloc[int(0.75*len(data)):,:]

print('Implementing Random Forest using training and testing data')

x = training_data['body']
y = training_data['subreddit']

# Vectorization of data using TF-ID vector
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
tfidf.fit(x)
traindata_tfid = tfidf.transform(x)
testdata_tfid = tfidf.transform(testing_data['body'])

# Random Forest
RF_trained_model = RandomForestClassifier(n_estimators=100)
RF_trained_model.fit(traindata_tfid,y)
print(RF_trained_model)

RF_predictions = RF_trained_model.predict(testdata_tfid)

from sklearn import metrics

print("Random Forest Accuracy:",metrics.accuracy_score(testing_data['subreddit'], RF_predictions))
print("Random Forest Confusion Matrix:",metrics.confusion_matrix(testing_data['subreddit'], RF_predictions))
print("Random Forest Classification Report:",metrics.classification_report(testing_data['subreddit'], RF_predictions))


print('Performing Feature Extraction on Development Data:')

# Removing Stop words
corpus = {'body':[]}
for word in development_data['body']:
    tokenized_subreddit = word_tokenize(word)
    for each in tokenized_subreddit:
        if each in stopwords.words('english'):
            tokenized_subreddit.remove(each)

    # Stemming
    stemmer = PorterStemmer()
    for i in range(len(tokenized_subreddit)):
        tokenized_subreddit[i] = stemmer.stem(tokenized_subreddit[i])
    tokenized_subreddit = ' '.join(tokenized_subreddit)
    corpus['body'].append(tokenized_subreddit)

corpus = pd.DataFrame(corpus)

# Transforming the data to lower case
corpus['body'] = corpus['body'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Removing Punctuations from the data
corpus['body'] = corpus['body'].str.replace('[^\w\s]','')

print('Implementing Random Forest using development and testing data')
x = corpus['body']
y = development_data['subreddit']

# Vectorization of data using TF-ID vector
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000,lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
tfidf.fit(x)
traindata_tfid = tfidf.transform(x)
testdata_tfid = tfidf.transform(testing_data['body'])

#Random Forest
RF_trained_model = RandomForestClassifier(n_estimators=100)
RF_trained_model.fit(traindata_tfid,y)
print(RF_trained_model)

RF_predictions = RF_trained_model.predict(testdata_tfid)

from sklearn import metrics

print("Random Forest Accuracy:",metrics.accuracy_score(testing_data['subreddit'], RF_predictions))
print("Random Forest Confusion Matrix:",metrics.confusion_matrix(testing_data['subreddit'], RF_predictions))
print("Random Forest Classification Report:",metrics.classification_report(testing_data['subreddit'], RF_predictions))


print('Performing Feature Extraction on training data set and then testing:')
# Removing Stop words
corpus = {'body':[]}
for word in training_data['body']:
    tokenized_subreddit = word_tokenize(word)
    for each in tokenized_subreddit:
        if each in my_stopwords:
            tokenized_subreddit.remove(each)

    # Stemming
    stemmer = PorterStemmer()
    for i in range(len(tokenized_subreddit)):
        tokenized_subreddit[i] = stemmer.stem(tokenized_subreddit[i])
    tokenized_subreddit = ' '.join(tokenized_subreddit)
    corpus['body'].append(tokenized_subreddit)

corpus = pd.DataFrame(corpus)

# Transforming the data to lower case
corpus['body'] = corpus['body'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Removing Punctuations from the data
corpus['body'] = corpus['body'].str.replace('[^\w\s]','')

x = corpus['body']
y = training_data['subreddit']

# Vectorization of data using TF-ID vector
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
tfidf.fit(x)
traindata_tfid = tfidf.transform(x)
testdata_tfid = tfidf.transform(testing_data['body'])

#Random Forest
RF_trained_model = RandomForestClassifier(n_estimators=100)
RF_trained_model.fit(traindata_tfid,y)
print(RF_trained_model)

RF_predictions = RF_trained_model.predict(testdata_tfid)

from sklearn import metrics

print("Random Forest Accuracy:",metrics.accuracy_score(testing_data['subreddit'], RF_predictions))
print("Random Forest Confusion Matrix:",metrics.confusion_matrix(testing_data['subreddit'], RF_predictions))
print("Random Forest Classification Report:",metrics.classification_report(testing_data['subreddit'], RF_predictions))