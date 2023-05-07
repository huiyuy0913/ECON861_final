import nltk
import pandas
import string
import numpy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import kfold_template
import kfold_template_part3
import dill as pickle
from sklearn.linear_model import LogisticRegression

dataset = pandas.read_csv("exam_data/part_3/training_data/dataset.csv")
# print(dataset)
# print(dataset.isna().sum())
# print(dataset.profile.value_counts())
# print(dataset.stars.value_counts())


'''Method 1: combining the review text and the profile picture types into one column'''
dataset['combined_text'] = dataset['reviewtext'] + ' ' + dataset['profile']
data = dataset['combined_text']
# data = dataset['reviewtext']
# data = dataset['profile']
# print(data)
target = dataset['stars']
lemmatizer = WordNetLemmatizer()

def pre_processing(text):
	text_processed = text.translate(str.maketrans('', '', string.punctuation))
	text_processed = text_processed.split()
	result = []
	for word in text_processed:
		word_processed = word.lower()
		if word_processed not in stopwords.words("english"):
			word_processed = lemmatizer.lemmatize(word_processed)
			result.append(word_processed)
	return result


count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

data = count_vectorize_transformer.transform(data)


machine = MultinomialNB() 
machine.fit(data,target)

results = kfold_template.run_kfold(data, target, machine, 4, False, True, True)
print("\n")
print([i[0] for i in results])
print("Test: The average accuracy score is", sum([i[0] for i in results])/len([i[0] for i in results]))
print("Train: The average accuracy score is", sum([i[1] for i in results])/len([i[1] for i in results]))




'''Method 2: train the model separately for different profile picture types'''

for profile in ['building','face', 'dog']:
	data = dataset[dataset.profile == profile]['reviewtext'].reset_index(drop=True)
	# print(data)

	target = dataset[dataset.profile == profile]['stars'].reset_index(drop=True)


	lemmatizer = WordNetLemmatizer()

	def pre_processing(text):
		text_processed = text.translate(str.maketrans('', '', string.punctuation))
		text_processed = text_processed.split()
		result = []
		for word in text_processed:
			word_processed = word.lower()
			if word_processed not in stopwords.words("english"):
				word_processed = lemmatizer.lemmatize(word_processed)
				result.append(word_processed)
		return result


	count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

	data = count_vectorize_transformer.transform(data)


	machine = MultinomialNB() 
	machine.fit(data,target)

	results = kfold_template.run_kfold(data, target, machine, 4, False, True, True)
	print("\n")
	print([i[0] for i in results])
	print("Test: The average accuracy score in "+ profile + " profile is", sum([i[0] for i in results])/len([i[0] for i in results]))
	print("Train: The average accuracy score in "+ profile + " profile is", sum([i[1] for i in results])/len([i[1] for i in results]))


'''Method 3: predict the quality of the programmer using the profile picture only, and supplement the prediction with the one I have done in part 1 using review text'''
data_reviewtext = dataset['reviewtext']
data_profile = dataset['profile']
# print(data_reviewtext)
# print(data_profile)
target = dataset['stars']
lemmatizer = WordNetLemmatizer()

def pre_processing(text):
	text_processed = text.translate(str.maketrans('', '', string.punctuation))
	text_processed = text_processed.split()
	result = []
	for word in text_processed:
		word_processed = word.lower()
		if word_processed not in stopwords.words("english"):
			word_processed = lemmatizer.lemmatize(word_processed)
			result.append(word_processed)
	return result


count_vectorize_transformer_reviewtext = CountVectorizer(analyzer=pre_processing).fit(data_reviewtext)
data_reviewtext = count_vectorize_transformer_reviewtext.transform(data_reviewtext)
machine_reviewtext = MultinomialNB() 
machine_reviewtext.fit(data_reviewtext,target)

count_vectorize_transformer_profile = CountVectorizer(analyzer=pre_processing).fit(data_profile)
data_profile = count_vectorize_transformer_profile .transform(data_profile)
machine_profile = MultinomialNB() 
machine_profile.fit(data_profile,target)


results = kfold_template_part3.run_kfold(data_reviewtext,data_profile, target, machine_reviewtext, machine_profile, 4, True)
print("\n")
print([i[0] for i in results])
print("Test: The average accuracy score is", sum([i[0] for i in results])/len([i[0] for i in results]))
print("Train: The average accuracy score is", sum([i[1] for i in results])/len([i[1] for i in results]))




dataset['combined_text'] = dataset['reviewtext'] + ' ' + dataset['profile']
data = dataset['combined_text']
# data = dataset['reviewtext']
# data = dataset['profile']
# print(data)
target = dataset['stars']
lemmatizer = WordNetLemmatizer()

def pre_processing(text):
	text_processed = text.translate(str.maketrans('', '', string.punctuation))
	text_processed = text_processed.split()
	result = []
	for word in text_processed:
		word_processed = word.lower()
		if word_processed not in stopwords.words("english"):
			word_processed = lemmatizer.lemmatize(word_processed)
			result.append(word_processed)
	return result


count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

data = count_vectorize_transformer.transform(data)

machine = MultinomialNB() 
machine.fit(data,target)

with open("machine_part3.pickle", "wb") as f:
	pickle.dump(machine, f)
	pickle.dump(count_vectorize_transformer, f)
	pickle.dump(lemmatizer, f)
	pickle.dump(stopwords, f)
	pickle.dump(string, f)
	pickle.dump(pre_processing, f)