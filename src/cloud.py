from wordcloud import WordCloud, STOPWORDS
import pandas as  pd
from pathlib import Path
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from clean import CleanText
nike_tweets = pd.read_csv(Path(__file__).absolute().parent.joinpath('../dataset/5000-justdoit-tweets-dataset/justdoit_tweets_2018_09_07_2.csv'))
nike_tweets = nike_tweets[['tweet_full_text']]
exclude = stopwords.words('english').append('https')
tweet_string = []
cleaner = CleanText()
words_to_exclude = {'https'}

for t in nike_tweets.tweet_full_text:
    
    tweet_string.append(t)
tweet_string = pd.Series(tweet_string).str.cat(sep=' ')
whitelist = ["n't", "not", "no"]
print(tweet_string)
print(stopwords.words('english'))
wc = WordCloud(width=1600, height=800,max_font_size=200,ranks_only="frequency",stopwords=STOPWORDS.union(words_to_exclude),collocations=False).generate(tweet_string)
plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()