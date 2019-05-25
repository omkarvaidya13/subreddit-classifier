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
import matplotlib.patches as mpatches
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

# Defining a list of stopwords
my_stopwords = stopwords.words('english')

def roc_curves(data):
    '''
    To generate ROC Curves for the data
    :param data:    data for which roc curve is to be generated
    :return:
    '''

    x = data['body']
    y = data['subreddit']
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=0,
                                                        shuffle=True)

    # Vectorization of data using TF-ID vector
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words='english',
                            ngram_range=(1, 1))
    tfidf.fit(x)
    traindata_tfid = tfidf.transform(train_x)
    testdata_tfid = tfidf.transform(test_x)

    # RANDOM FOREST
    RF_trained_model = RandomForestClassifier(n_estimators=100)
    RF_trained_model.fit(traindata_tfid, train_y)

    RF_probs = RF_trained_model.predict_proba(testdata_tfid)[:, 1]

    from sklearn.metrics import roc_auc_score

    # Calculate roc auc score
    roc_value = roc_auc_score(test_y, RF_probs)

    from sklearn.metrics import roc_curve
    base_fpr, base_tpr, _ = roc_curve(test_y, [1 for _ in range(len(test_y))])
    model_fpr, model_tpr, _ = roc_curve(test_y, RF_probs)

    return base_fpr,base_tpr,model_fpr,model_tpr,roc_value


def main():
    # Connecting to the reddit script app through PRAW API
    reddit = praw.Reddit(client_id='Nuhb2V_ONfbRAg', \
                         client_secret='ao8oVLq-ZEz1a-9leqg1FqGdCIg', \
                         user_agent='redditApp1', \
                         username='ketand2017', \
                         password='passwordfis'
                         )

    # The subreddits from which data is collected
    subreddit1 = reddit.subreddit('computerscience')
    subreddit2 = reddit.subreddit('datascience')
    subreddit3 = reddit.subreddit('GRE')

    # Collecting the latest 500 posts from each subreddit
    new_subreddit1 = subreddit1.new(limit=500)
    new_subreddit2 = subreddit2.new(limit=500)
    new_subreddit3 = subreddit3.new(limit=500)

    # Storing the subreddits data into json format
    first_subreddit_data = {"body": [], "subreddit": []}
    second_subreddit_data = {"body": [], "subreddit": []}
    third_subreddit_data = {"body": [], "subreddit": []}

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

    # Merging first and second subreddit data
    first_second_data = data1.append(data2)
    first_second_data['subreddit'].replace({'computerscience': 0, 'datascience': 1}, inplace=True)

    # Merging second and third subreddit data
    second_third_data = data2.append(data3)
    second_third_data['subreddit'].replace({'datascience': 0, 'GRE': 1}, inplace=True)

    # Merging first and third subreddit data
    first_third_data = data1.append(data3)
    first_third_data['subreddit'].replace({'computerscience': 0, 'GRE': 1}, inplace=True)

    #Function call for first and second subreddit
    base_fpr, base_tpr, model_fpr, model_tpr,roc_value = roc_curves(first_second_data)
    print("computerscience and datascience ROC value:", roc_value)

    plt.figure(figsize=(16, 6))
    plt.rcParams['font.size'] = 12

    plt.subplot(1,3,1)
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - computerscience and datascience')

    # Function call for second and third subreddit
    base_fpr, base_tpr, model_fpr, model_tpr,roc_value = roc_curves(second_third_data)
    print("datascience and GRE ROC value:", roc_value)

    plt.subplot(1, 3, 2)
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - datascience and GRE')

    # Function call for first and third subreddit
    base_fpr, base_tpr, model_fpr, model_tpr,roc_value = roc_curves(first_third_data)
    print("computerscience and GRE ROC value:", roc_value)

    plt.subplot(1, 3, 3)
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - computerscience and GRE')

    plt.show()

if __name__ == '__main__':
    main()