import re
import sys
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
import string
from nltk.corpus import stopwords

class CleanText(BaseEstimator, TransformerMixin):

    def preprocess_word(self,word):
        # Remove punctuation
        # Convert more than 2 letter repetitions to 2 letter
        # funnnnny --> funny
        word = re.sub(r'(.)\1+', r'\1\1', word)
        # Remove - & '
        word = re.sub(r'(-|\')', '', word)
        return word

    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)

    def is_valid_word(self,word):
        # Check if word begins with an alphabet
        return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


    def handle_emojis(self,tweet):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', 'EMOPOS', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', 'EMOPOS', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', 'EMOPOS', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', 'EMOPOS', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', 'EMONEG', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', 'EMONEG', tweet)
        return tweet
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words)

    def preprocess_tweet(self,tweet):
        processed_tweet = []
        # Convert to lower case
        tweet = tweet.lower()
        # Replaces URLs with the word URL
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
        # Replace @handle with the word USER_MENTION
        tweet = re.sub(r'@[\S]+', '', tweet)
        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        # Remove RT (retweet)
        tweet = re.sub(r'\brt\b', '', tweet)
        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', '', tweet)
        # Strip space, " and ' from tweet
        tweet = tweet.strip(' "\'')
        # Replace emojis with either EMO_POS or EMO_NEG
        tweet = self.handle_emojis(tweet)
        # Replace multiple spaces with a single space
        tweet = re.sub(r'\s+', ' ', tweet)

        tweet = self.remove_digits(tweet)
        tweet = self.remove_punctuation(tweet)
        tweet = re.sub(' +',' ',tweet)
        tweet = self.remove_stopwords(tweet)

        #words = tweet.split()
       # porter = PorterStemmer()
        #for word in words:
       #     word = self.preprocess_word(word)
        #    if self.is_valid_word(word):
        ##        word = str(porter.stem(word))
         #       processed_tweet.append(word)

        #return ' '.join(processed_tweet)
        return tweet
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.preprocess_tweet)
        return clean_X    