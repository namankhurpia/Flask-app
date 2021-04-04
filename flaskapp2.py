from flask import Flask, request, jsonify
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 20)


app = Flask(__name__)

@app.route('/')
def hello_world():
	return 'Hello my fellow humannsss!!!'

@app.route('/getlan',methods = ['POST'])
def getlang():
	dataset = pd.read_csv('dataset.tsv', delimiter = '\t', quoting = 3)
	corpus = []
	for i in range(0, 20):
	  review =   dataset['code'][i]
	  #review = review.lower()
	  review = review.split()
	  ps = PorterStemmer()
	  all_stopwords = stopwords.words('english')
	  #all_stopwords.remove('not')
	  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
	  review = ' '.join(review)
	  corpus.append(review)
	  
	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer(max_features = 20)
	X = cv.fit_transform(corpus).toarray()
	y = dataset.iloc[:, -1].values

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


	from sklearn.naive_bayes import GaussianNB
	classifier = GaussianNB()
	classifier.fit(X_train, y_train)

	y_pred = classifier.predict(X_test)
	#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

	from sklearn.metrics import confusion_matrix, accuracy_score
	cm = confusion_matrix(y_test, y_pred)
	#print(cm)
	accuracy_score(y_test, y_pred)
	#print(accuracy_score(y_test , y_pred))


	x= request.get_json(force = True)
	text = x['data'] 
	sent = text
	#sent = train(text)
	#sent = sent[0]
	new_review = sent 
	print(sent)
	#new_review = 'class main { public static void main ( String []args) { System.out.print(hello world) ; } } '

	new_review = new_review.lower()
	new_review = new_review.split()
	ps = PorterStemmer()
	all_stopwords = stopwords.words('english')
	#all_stopwords.remove('not')
	new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
	new_review = ' '.join(new_review)
	new_corpus = [new_review]
	new_X_test = cv.transform(new_corpus).toarray()
	#print("array is" ,new_X_test)
	new_y_pred = classifier.predict(new_X_test)
	#print("result is - ")
	print(new_y_pred)
	return np.array_str(new_y_pred)

if __name__ == "__main__":
	app.run(use_reloader = True)




