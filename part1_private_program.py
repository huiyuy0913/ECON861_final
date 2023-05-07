import nltk
import pandas
import string
import numpy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import kfold_template

import dill as pickle

dataset = pandas.read_csv("exam_data/part_1/training_data/dataset.csv")
# print(dataset)
# print(dataset.isna().sum())
# print(dataset.stars.value_counts())

data = dataset['reviewtext']
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

print([i[0] for i in results])
print("The average accuracy score is", sum([i[0] for i in results])/len([i[0] for i in results]))
print("The average accuracy score is", sum([i[1] for i in results])/len([i[1] for i in results]))



machine = MultinomialNB() 
machine.fit(data,target)

with open("machine.pickle", "wb") as f:
	pickle.dump(machine, f)
	pickle.dump(count_vectorize_transformer, f)
	pickle.dump(lemmatizer, f)
	pickle.dump(stopwords, f)
	pickle.dump(string, f)
	pickle.dump(pre_processing, f)
