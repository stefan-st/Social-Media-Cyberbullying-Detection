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
from xlwt import Workbook, Worksheet 
import pickle
import tkinter
import sys

FORMSPING_MAX_LEN = 75
TWITTER_MAX_LEN = 30
USER_MAX = 5
LINK_MAX = 5
HASHTAG_MAX = 5


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
	completness = ["everyday", "everybody", "all", "always", "never", "nobody"
	"everytime", "none", "nothing"]

	mention_features = get_mention_features(post)

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

	nr_swears /= nr_words
	nr_first_person /= nr_words
	mention_features[0] /= nr_words
	mention_features[1] /= nr_words

	post_len = nr_words / max_len
	if post_len > 1:
		post_len = 1

	return np.array([nr_swears, capital_letters, comp_feat, post_len, mention_features[1], nr_second_person, personal_swear, mention_features[0]])

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

q = []
score = []

options = ["Bullying", "Maybe Bullying", "Not Bullying"]

a = []

class Quiz:
    def __init__(self, master):
        self.opt_selected = tkinter.IntVar()
        self.qn = 0
        self.correct = 0
        self.ques = self.create_q(master, self.qn)
        self.opts = self.create_options(master, 3)
        self.display_q(self.qn)
        self.button = tkinter.Button(master, text="Finish", command=master.destroy)
        self.button.pack(side=tkinter.BOTTOM)
        self.button = tkinter.Button(master, text="Next", command=self.next_btn)
        self.button.pack(side=tkinter.BOTTOM)

    def create_q(self, master, qn):
        w = tkinter.Label(master, text=q[qn])
        w.pack(side=tkinter.TOP)
        return w

    def create_options(self, master, n):
        b_val = 0
        b = []
        while b_val < n:
            btn = tkinter.Radiobutton(master, text="foo", variable=self.opt_selected, value=b_val+1)
            b.append(btn)
            btn.pack(side=tkinter.TOP, anchor="w")
            b_val = b_val + 1
        return b

    def display_q(self, qn):
        b_val = 0
        self.opt_selected.set(0)
        char_list = [q[qn][j] for j in range(len(q[qn])) if ord(q[qn][j]) in range(65536)]
        tweet = ''
        for j in char_list:
            tweet = tweet + j
        self.ques['text'] = tweet
        for op in options:
            self.opts[b_val]['text'] = op
            b_val = b_val + 1

    def store_q(self, qn):
        a.append(self.opt_selected.get())

    def print_results(self):
        print("Score: ", self.correct, "/", len(q))

    def next_btn(self):
        self.store_q(self.qn)
        self.qn = self.qn + 1
        if self.qn > len(q):
            self.qn = len(q) - 1
        self.display_q(self.qn)


if __name__ == '__main__':

	#load list of swear words
	swear_words = load_swear_words()

	# load the trained models used for classification
	clf_agg = pickle.load(open('models/agg_mode.sav', 'rb'))
	vectorizer_agg = pickle.load(open('models/agg_vect.pickle', 'rb'))
	tfidf_transformer_agg = pickle.load(open('models/tfidf_agg_vect.pickle', 'rb'))

	clf_bully = pickle.load(open('models/bully_model_formspring.sav', 'rb'))
	vectorizer_bully = pickle.load(open('models/bully_vect_formspring.pickle', 'rb'))
	tfidf_transformer_bully = pickle.load(open('models/tfidf_bully_vect_formspring.pickle', 'rb'))
	
	clf_bully_t = pickle.load(open('models/bully_model_twitter.sav', 'rb'))
	vectorizer_bully_t = pickle.load(open('models/bully_vect_twitter.pickle', 'rb'))
	tfidf_transformer_bully_t = pickle.load(open('models/tfidf_bully_vect_twitter.pickle', 'rb'))
	
	# load new set of posts to classify
	X_new = load_tweets(sys.argv[1])
	X_new = process_tweets_simple(X_new)

	X_new_clean = [prepare_text(post) for post in X_new]
	X_add_features_new = np.array([process_post(X_new[i], swear_words) for i in range(len(X_new))])


	# compute the feature values for the classification
	X_new_features = vectorizer_bully.transform(X_new_clean)
	X_new_features = tfidf_transformer_bully.fit_transform(X_new_features)

	X_features = vectorizer_agg.transform(X_new_clean)
	X_features= tfidf_transformer_agg.transform(X_features)
	y_pred_agg = clf_agg.predict_proba(X_features)
	y_pred_agg = np.array([x[1] for x in y_pred_agg])

	X_new_test = hstack((y_pred_agg[:, None], X_add_features_new, X_new_features))

	# get the prediction from the 2 classifiers
	y_pred = clf_bully.predict_proba(X_new_test)
	y_pred_t = clf_bully_t.predict_proba(X_new_test)

	# get the final prediction by combinig the results from
	# the 2 classifiers
	y_pred = (y_pred + y_pred_t) / 2
	y_pred = [x[1] for x in y_pred]

	texts = [post[1:] for post in X_new]

	labels, texts = (list(t) for t in zip(*sorted(zip(y_pred, texts), reverse=True)))

	# write all the classified posts in a sheet
	wb = Workbook() 
	for idx, text in enumerate(texts):
		q.append(text)
		score.append(labels[idx])

	sheet1 = wb.add_sheet('Sheet 1') 
	sheet1.col(0).width = 256 * 140
	for idx, label in enumerate(y_pred):
			sheet1.write(idx, 0, texts[idx])
			sheet1.write(idx, 1, labels[idx])

	wb.save('results/demo.xls') 

	root = tkinter.Tk()
	root.geometry("1000x200")
	app = Quiz(root)
	root.mainloop()

	hlabels = []
	for val in a:
		if val == 1:
			hlabels.append("Bullying")
		if val == 2:
			hlabels.append("Maybe Bullying")
		if val == 3:
			hlabels.append("Not Bullying")

	# write the human classified posts in a sheet
	wb = Workbook() 

	sheet1 = wb.add_sheet('Sheet 1') 
	sheet1.col(0).width = 256 * 140
	sheet1.col(2).width = 256 * 20
	for idx in range(len(a)):
			sheet1.write(idx, 0, texts[idx])
			sheet1.write(idx, 1, labels[idx])
			sheet1.write(idx, 2, hlabels[idx])

	wb.save('results/demo_hl.xls') 
