import csv
import pandas as pd
import string
import re
import numpy as np
import random
import json
import codecs
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from scipy.sparse import hstack, vstack
import xlwt 
from xlwt import Workbook 
import pickle

FORMSPING_MAX_LEN = 75
TWITTER_MAX_LEN = 30
USER_MAX = 5
LINK_MAX = 5
HASHTAG_MAX = 5

"""
This methods loads the formspring dataset
"""
def load_formspring_data():
	posts = pd.read_csv('formspring_data.csv', sep = '\t')
	pd.set_option('display.max_columns', 20)
	pd.set_option('display.max_colwidth', 500)

	posts['label'] = np.array(posts.apply(get_label, axis=1))
	# posts['post'] = np.array(posts.apply(get_post_text, axis=1))
	posts['post'] = np.array(posts.apply(clean_text, axis=1))

	posts_text = posts['post'].to_list()
	y = posts['label'].to_list()

	bully = []
	y_bully = []
	post_lengths = {}

	for idx, post in enumerate(posts_text):
		post_len = len(post.split())

		post_lengths[post_len] = post_lengths.get(post_len, 0) + 1

		if y[idx] == 1:
			if random.random() < 0.8:
				bully.append('F' + post)
				y_bully.append(1)

	X = posts_text

	for i in range(0):
		X = np.concatenate((X, bully))
		y = np.concatenate((y, y_bully))

	return y, X, post_lengths

"""
Loads Twitter posts from a json file
"""
def load_tweets(filename):
	datastore = []
	with codecs.open(filename, 'r') as inputFile:
		for line in inputFile:
			datastore.append(json.loads(line))
	return datastore

"""
Loads the Twitter dataset
"""
def load_twitter_data():
	data = []
	with open('data.csv', 'r') as csvFile:
		reader = csv.reader(csvFile)
		for row in reader:
			data.append(row)

	labels = {row[0]:1 if row[2] == 'y' else -1 for row in data}

	X = load_tweets('tweets.json')
	y = [labels[tweet['id_str']] for tweet in X]

	return y, X

"""
Method to get the label based on the level of agrressivity from
the Formspring dataset
"""
def get_label(post):
	aggresivity = 0
	if post.ans1 == 'Yes':
		aggresivity += 1
	if post.ans2 == 'Yes':
		aggresivity += 1
	if post.ans3 == 'Yes':
		aggresivity += 1

	if aggresivity >= 2:
		return 1
	return -1

"""
Load the aggresivity labelled dataset from Colins
"""
def load_colins_data(filename):
	X = []
	y = []

	with open(filename, 'r', encoding='utf-8') as csvFile:
		reader = csv.reader(csvFile)
		for row in reader:
			X.append(row[1])
			if row[2] == 'NAG':
				y.append(-1)
			else:
				y.append(1)

	return np.array(X), np.array(y)

"""
Returns a classifier based on the type
"""
def get_classifier(type):
	if type == 'random_forest':
		return RandomForestClassifier(n_estimators=100)
	if type == 'svm_W':
		return svm.SVC(kernel='linear', class_weight={1: 100})
	if type == 'svm':
		return svm.SVC(kernel='linear', class_weight='balanced', probability=True)
	if type == 'lr':
		return LogisticRegression()

"""
Preprocess a Twitter post
"""
def process_tweet_text(tweet_text):
	new_text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', 'HTTPLINK', tweet_text)
	new_text = re.sub(r'@(\w|\.|\/|\?|\=|\&|\%)*\b', '@USERNAME', new_text)

	post_text = re.sub(r"""
               [,.;#?!&$]+  # Accept one or more copies of punctuation
               \ *          # plus zero or more copies of a space,
               """,
               " ",         # and replace it with a single space
               new_text, flags=re.VERBOSE)
	return post_text

"""
Preprocess the tweets in the Twitter dataset
"""
def process_tweets(tweets):
	y = []

	X_text = []
	data = []
	post_lengths = {}

	with open('data.csv', 'r') as csvFile:
		reader = csv.reader(csvFile)
		for row in reader:
			data.append(row)

	labels = {row[0]:1 if row[2] == 'y' else -1 for row in data}

	for id, tweet in enumerate(tweets):
		post_len = len(tweet['text'].split())

		post_lengths[post_len] = post_lengths.get(post_len, 0) + 1

		if tweet['lang'] == 'en':
			X_text.append('T' + tweet['text'])
			y.append(labels[tweet['id_str']])

	return y, X_text, post_lengths

def process_tweets_simple(tweets):
	X_text = []
	ids = []
	for idx, tweet in enumerate(tweets):
		#new_text = process_tweet_text(tweet['text'])
		if tweet['id_str'] not in ids:
			new_text = 'T' + tweet['text']
			X_text.append(new_text)
			ids.append(tweet['id_str'])

	return X_text

"""
Preprocess text in the Formspring dataset
"""
def clean_text(post):
	post_text = post.post

	new_text = 'F' + post_text.replace('Q: ', ' ').replace('A: ', ' ').replace('<br>', ' ').replace('&#039;', '\'').replace('&quot;', ' ').replace('&#39;', '\'')
	return new_text

def get_post_text(post):
	return post.post

def clean_text_simple(post_text):
	post_text = post_text.replace('Q: ', ' ').replace('A: ', ' ').replace('<br>', ' ').replace('&#039;', '\'').replace('&quot;', ' ')

	return post_text

"""
Method used for getting the scores for each type of classifier
"""
def classify_results(y_train, X_train, y_test, X_test, dataset_type):
	vectorizer = CountVectorizer(stop_words='english', ngram_range = (1, 2), max_features=10000)
	tfidf_transformer = TfidfTransformer(norm = 'l2')
	X_train = vectorizer.fit_transform(X_train)

	vocab_size = len(vectorizer.vocabulary_)
	X_test = vectorizer.transform(X_test)
	X_train = tfidf_transformer.fit_transform(X_train)
	X_test = tfidf_transformer.transform(X_test)

	classifiers = ['svm_W', 'svm', 'random_forest', 'lr']

	print(dataset_type, '\n')

	for c in classifiers:
		print(c)
		clf = get_classifier(c)
		clf.fit(X_train, y_train)
		
		y_pred = clf.predict(X_test)
		print("Precision: ", precision_score(y_test, y_pred))
		print("Recall: ", recall_score(y_test, y_pred))
		print("F1-Score: ", f1_score(y_test, y_pred))
		print("Accuracy: ", accuracy_score(y_test, y_pred))

		plot_confusion_matrix(y_test, y_pred, [0, 1], normalize=False)

		print('\n')

"""
Loads the list of swear words
"""
def load_swear_words():
	swear_words = []
	with open('swear_words.txt') as inFile:
		for word in inFile:
			swear_words.append(word[:-1])
	return swear_words

"""
Compute the engineered features values for a post
"""
def process_post(post, swear_words):

	nr_swears = 0
	nr_words = 0
	nr_second_person = 0
	nr_first_person = 0

	if post[0] == 'T':
		max_len = TWITTER_MAX_LEN
	else:
		max_len = FORMSPING_MAX_LEN

	post = post[1:]

	table = str.maketrans('', '', string.punctuation)
	second_person = ["you", "your", "yours", "you're", "youre", "yo"]
	first_person = ["i", "im", "i'm", "am"]
	completness = ["very", "most", "least", "everyday", "everybody", "all", "always", "never", "nobody"
	"everytime", "none", "nothing"]

	mention_features = get_mention_features(post)
	user_mention = mention_features[0]
	hyperlinks = mention_features[1]
	hashtags = mention_features[2]

	word_list = post.split(" ")
	word_triplets = [word_list[i:i + 3] for i in range(len(word_list) - 2)]

	personal_swear = 0
	for triplet in word_triplets:
		is_personal = False
		is_vulgar = False
		triplet = [word.translate(table).lower() for word in triplet]
		if triplet[0] in second_person or triplet[1] in second_person or triplet[2] in second_person:
			is_personal = True
		if triplet[0].startswith("@") or  triplet[1].startswith("@") or triplet[2].startswith("@"):
			is_personal = True
		if triplet[0] in swear_words or triplet[1] in swear_words or triplet[2] in swear_words:
			is_vulgar = True
		if is_personal and is_vulgar:
			personal_swear = 1

	stripped_text = re.sub(r'@(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', post)
	stripped_text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', stripped_text)

	capital_letters = sum(1 for c in stripped_text if c.isupper())

	capital_letters /= len(stripped_text)
	
	word_list = stripped_text.split(" ")
	comp_feat = 0

	for word in stripped_text.split(" "):
		nr_words += 1
		if word.translate(table).lower() in swear_words:
			nr_swears += 1

		if word.translate(table).lower() in second_person:
			nr_second_person = 1

		if word.translate(table).lower() in first_person:
			nr_first_person += 1

		if word.translate(table).lower() in completness:
			comp_feat = 1

	swears_present = 1 if nr_swears > 0 else 0
	nr_swears /= nr_words
	nr_first_person /= nr_words
	mention_features[0] /= nr_words
	mention_features[1] /= nr_words

	post_len = nr_words / max_len
	if post_len > 1:
		post_len = 1

	content_features = [nr_swears, swears_present, hyperlinks, capital_letters, comp_feat, post_len]
	subjectivity_features = [nr_second_person, nr_first_person, user_mention, personal_swear]
	return np.array(content_features + subjectivity_features)

def prepare_text(post):
	post = post[1:].lower()
	stripped_text = re.sub(r'(\&)\w+(\;)', '', post)
	stripped_text = re.sub(r'@(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', stripped_text)
	stripped_text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', stripped_text)
	stripped_text = re.sub(r'[^\w\s]|(.)(?=\1{2,})', '', stripped_text)
	clean_text = ""

	table = str.maketrans('', '', ',.;?!')
	for word in stripped_text.split():
		clean_text += word.translate(table) + " "

	return clean_text

def clean_dataset(X):
	return [prepare_text(post) for post in X]

"""
Compute the mentions + links features for a post
"""
def get_mention_features(post):
	matches_users = re.findall(r'@(\w|\.|\/|\?|\=|\&|\%)*\b', post)
	matches_links = re.findall(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', post)
	matches_hashtags = re.findall(r'#(\w|\.|\/|\?|\=|\&|\%)*\b', post)

	user_mention = len(matches_users) / USER_MAX
	link_mention = len(matches_links) / LINK_MAX
	hashtags = len(matches_hashtags) / HASHTAG_MAX

	return [user_mention, link_mention, hashtags]

"""
Train a classifier on the given dataset
"""
def train_classifier(X_train, y_train, classifier):
	vectorizer = CountVectorizer(stop_words='english', ngram_range = (1, 3), max_features=10000)
	tfidf_transformer = TfidfTransformer(norm = 'l2')
	X_train = vectorizer.fit_transform(X_train)
	X_train = tfidf_transformer.fit_transform(X_train)


	clf = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
	clf.fit(X_train, y_train)

	return clf, vectorizer, tfidf_transformer

def train_test_classifier(X_train, X_test, y_train, y_test, clf):
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)

	print("Precision: ", precision_score(y_test, y_pred))
	print("Recall: ", recall_score(y_test, y_pred))
	print("F1-Score: ", f1_score(y_test, y_pred))
	print("Accuracy: ", accuracy_score(y_test, y_pred))

	plot_confusion_matrix(y_test, y_pred, [0, 1], normalize=False)

	return clf

def test_combined(X_test, y_test, clf1, clf2):

	y_pred1 = clf1.predict_proba(X_test)
	y_pred2 = clf2.predict_proba(X_test)

	y_pred = (y_pred1 + y_pred2) / 2
	y_pred = [1 if x[1] > 0.5 else -1 for x in y_pred]

	print("Precision: ", precision_score(y_test, y_pred))
	print("Recall: ", recall_score(y_test, y_pred))
	print("F1-Score: ", f1_score(y_test, y_pred))
	print("Accuracy: ", accuracy_score(y_test, y_pred))

def load_train_data():
	# load formspring dataset
	y_formspring, X_formspring, X_lengths = load_formspring_data()

	# load twitter dataset
	y_twitter, X_twitter = load_twitter_data()
	y_twitter, X_twitter, X_lengths = process_tweets(X_twitter)

	print('Formspring: ', sum(y_formspring), len(y_formspring))
	print('Twitter: ', sum(y_twitter), len(y_twitter))

	# dataset for training and testing the bullying classifier
	X_ = X_formspring #+ X_twitter
	y_ = y_formspring #+ y_twitter

	return X_formspring, X_twitter, y_formspring, y_twitter

def get_train_test_features(X_train, X_test, experiment = 'A'):
	# load list of swear words
	swear_words = load_swear_words()

	# get additional computed features
	X_add_features_train = np.array([process_post(X_train[i], swear_words) for i in range(len(X_train))])
	X_add_features_test = np.array([process_post(X_test[i], swear_words) for i in range(len(X_test))])

	# clean dataset before computing TF-IDF
	X_train_clean = clean_dataset(X_train)
	X_test_clean = clean_dataset(X_test)

	# TF-IDF transformer for twitter posts
	vectorizer_bully = CountVectorizer(stop_words='english', ngram_range = (1, 2), max_features=5000)
	tfidf_transformer_bully = TfidfTransformer(norm = 'l2')
	X_train_features = vectorizer_bully.fit_transform(X_train_clean)
	X_train_features = tfidf_transformer_bully.fit_transform(X_train_features)

	X_test_features = vectorizer_bully.transform(X_test_clean)
	X_test_features = tfidf_transformer_bully.fit_transform(X_test_features)

	# get the tfidf features for predicting the aggresivness of the post
	X_tfid_bully_train = vectorizer_agg.transform(X_train_clean)
	X_tfid_bully_train = tfidf_transformer_agg.transform(X_tfid_bully_train)

	X_tfid_bully_test = vectorizer_agg.transform(X_test_clean)
	X_tfid_bully_test = tfidf_transformer_agg.transform(X_tfid_bully_test)

	# aggresivity feature
	y_pred_agg_train = clf_agg.predict_proba(X_tfid_bully_train)
	y_pred_agg_train = np.array([[x[1]] for x in y_pred_agg_train])

	y_pred_agg_test = clf_agg.predict_proba(X_tfid_bully_test)
	y_pred_agg_test = np.array([[x[1]] for x in y_pred_agg_test])

	content_features_train = X_add_features_train[:, :6]
	subjectivity_features_train = X_add_features_train[:, 6:]

	content_features_test = X_add_features_test[:, :6]
	subjectivity_features_test = X_add_features_test[:, 6:]

	train_feat = None
	test_feat = None
	if experiment == 'A':
		train_feat = X_train_features
		test_feat = X_test_features
	elif experiment == 'B':
		train_feat = hstack((content_features_train, X_train_features))
		test_feat = hstack((content_features_test, X_test_features))
	elif experiment == 'C':
		train_feat = hstack((content_features_train, subjectivity_features_train, X_train_features))
		test_feat = hstack((content_features_test, subjectivity_features_test, X_test_features))
	elif experiment == 'D':
		train_feat = hstack((y_pred_agg_train, content_features_train, subjectivity_features_train, X_train_features))
		test_feat = hstack((y_pred_agg_test, content_features_test, subjectivity_features_test, X_test_features))
	elif experiment == 'E':
		train_feat = hstack((y_pred_agg_train, X_train_features))
		test_feat = hstack((y_pred_agg_test, X_test_features))
	elif experiment == 'F':
		train_feat = hstack((y_pred_agg_train, content_features_train, subjectivity_features_train))
		test_feat = hstack((y_pred_agg_test, content_features_test, subjectivity_features_test))

	return train_feat, test_feat

if __name__ == '__main__':

	# load list of swear words
	swear_words = load_swear_words()

	# load aggresivity trained model
	clf_agg = pickle.load(open('models/agg_mode.sav', 'rb'))
	vectorizer_agg = pickle.load(open('models/agg_vect.pickle', 'rb'))
	tfidf_transformer_agg = pickle.load(open('models/tfidf_agg_vect.pickle', 'rb'))


	# load training data from Twitter and Formspring
	X_, X_t, y_, y_t = load_train_data()

	X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.001, shuffle=True)
	X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size = 0.001, shuffle=True)
	
	print('Compute the features for the train sets')
	# get the train/test features for the wanted setting
	X_train, X_test = get_train_test_features(X_train, X_test, 'D')
	X_train_t, X_test_t = get_train_test_features(X_train_t, X_test_t, 'D')


	print('Train model')
	kernel_type = ['linear', 'poly', 'rbf', 'sigmoid']
	C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
	bully_weight = [1., 5.,10., 15., 20.]

	# Only twitter data: kernel=linear, C=1, weight=1.15
	class_weight = {0: 1., 1: 5}
	clf_bully = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
	clf_bully = train_test_classifier(X_train, X_test, y_train, y_test, clf_bully)


	class_weight = {0: 1., 1: 1.15}
	clf_bully_t = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
	clf_bully_t = train_test_classifier(X_train_t, X_test_t, y_train_t, y_test_t, clf_bully_t)

	test_combined(X_test, y_test, clf_bully, clf_bully_t)

	pickle.dump(clf_bully, open('models/bully_model_formspring.pkl', 'wb'))
	pickle.dump(clf_bully_t, open('models/bully_model_twitter.pkl', 'wb'))
	# pickle.dump(vectorizer_bully, open('models/bully_vect_formspring.pickle', 'wb'))
	# pickle.dump(tfidf_transformer_bully, open('models/tfidf_bully_vect_formspring.pickle', 'wb'))
