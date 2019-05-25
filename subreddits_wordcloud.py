import praw
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from autocorrect import spell
from nltk.stem import WordNetLemmatizer
import matplotlib.patches as mpatches
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Defining a list of stopwords
my_stopwords = stopwords.words('english')
my_stopwords.append('would')

def wordcloud(data):
    '''
    To generate Word Cloud for the data
    :param data:    data for which word cloud is to be generated
    :return:
    '''

    # For first subreddit
    text1 = ''.join(x for x in data['body'][:499])

    # For second subreddit
    text2 = ''.join(x for x in data['body'][500:1000])

    # For third subreddit
    text3 = ''.join(x for x in data['body'][1000:])

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.size'] = 12


    wordcloud1 = WordCloud().generate(text1)
    plt.subplot(1,3,1)
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.axis("off")
    plt.title('computerscience Wordcloud')

    wordcloud2 = WordCloud().generate(text2)
    plt.subplot(1,3,2)
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")
    plt.title('datascience Wordcloud')

    wordcloud3 = WordCloud().generate(text3)
    plt.subplot(1,3,3)
    plt.imshow(wordcloud3, interpolation='bilinear')
    plt.axis("off")
    plt.title('GRE Wordcloud')

    plt.show()

def main():
    print('Generating Wordcloud...')

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

    # Merging all the data
    data = data1.append(data2)
    data = data.append(data3)

    # Removing Stop words
    corpus = {'body':[]}
    for word in data['body']:
        tokenized_subreddit = word_tokenize(word)
        for each in tokenized_subreddit:
            if each in my_stopwords:
                tokenized_subreddit.remove(each)
        tokenized_subreddit = ' '.join(tokenized_subreddit)
        corpus['body'].append(tokenized_subreddit)

    corpus = pd.DataFrame(corpus)

    # Transforming the data to lower case
    corpus['body'] = corpus['body'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Removing Punctuations from the data
    corpus['body'] = corpus['body'].str.replace('[^\w\s]','')

    #Calling the function wordcloud to plot the wordcloud.
    wordcloud(corpus)
    print('Done')

if __name__ == '__main__':
    main()