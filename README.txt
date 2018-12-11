instructions:

1. Run setup.py. This will install all neccesary libraries along with the nltk stopwords corpus.
2. Run preprocess.py. the preproccessed training and testing data can be found in the dataset folder.
3. Run train.py. This will train the model, save the model to disk to be loaded for predictions, and output the test results to the ouput folder.
4. Run preproccess-nike.py. This will preprocess the dataset from the Nike #justdoit campaign and save it in the dataset folder.
5. Run predict.py. This will predict the sentiment of each tweet, along with the probability of it being the correct sentiment, and save it to the output folder.




Information: 


	This is the final system for analyzing the 5000 #justdoit tweets dataset provided on Kaggle. 
	
	Credits: Ameer Akashe, Shivam Bhattacharya, Mark Maatouk


	The original training and testing dataset can be found at http://alt.qcri.org/semeval2017/task4/


	The Nike twitter dataset can be found at https://www.kaggle.com/eliasdabbas/5000-justdoit-tweets-dataset