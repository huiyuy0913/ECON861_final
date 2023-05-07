import nltk
import pandas
import string
import numpy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import kfold_template

import dill as pickle

dataset = pandas.read_csv("exam_data/part_1/training_data/dataset.csv")

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

# print("\n\n")
# for i in range(2,6):
# 	results = kfold_template.run_kfold(data, target, machine, i, False, True, False)
# 	print("Test: The average accuracy score for Naive Bayesian is", sum([i[0] for i in results])/len([i[0] for i in results]))
# 	print("Train: The average accuracy score for Naive Bayesian is", sum([i[1] for i in results])/len([i[1] for i in results]))

results = kfold_template.run_kfold(data, target, machine, 4, False, True, True)
print("\n\n")
print([i[0] for i in results])
print([i[2] for i in results])
print("Test: The average accuracy score for Naive Bayesian is", sum([i[0] for i in results])/len([i[0] for i in results]))
print("Train: The average accuracy score for Naive Bayesian is", sum([i[1] for i in results])/len([i[1] for i in results]))


machine = LogisticRegression(penalty='l1',multi_class='multinomial', solver='saga',max_iter=20000) 
# machine = LogisticRegression(multi_class='multinomial', solver='newton-cg') 
machine.fit(data,target)

# print("\n\n")
# for i in range(2,6):
# 	results = kfold_template.run_kfold(data, target, machine, i, False, True, False)
# 	print("Test: The average accuracy score for Logistic Regression is", sum([i[0] for i in results])/len([i[0] for i in results]))
# 	print("Train: The average accuracy score for Logistic Regression is", sum([i[1] for i in results])/len([i[1] for i in results]))


results = kfold_template.run_kfold(data, target, machine, 4, False, True, True)
print("\n\n")
print([i[0] for i in results])
print([i[2] for i in results])
print("Test: The average accuracy score for Logistic Regression is", sum([i[0] for i in results])/len([i[0] for i in results]))
print("Train: The average accuracy score for Logistic Regression is", sum([i[1] for i in results])/len([i[1] for i in results]))



machine = RandomForestClassifier() 
machine.fit(data,target)
results = kfold_template.run_kfold(data, target, machine, 4, False, True, True)
print("\n\n")
print([i[0] for i in results])
print([i[2] for i in results])
print("Test: The average accuracy score for Random Forest is", sum([i[0] for i in results])/len([i[0] for i in results]))
print("Train: The average accuracy score for Random Forest is", sum([i[1] for i in results])/len([i[1] for i in results]))


print("\n\n")


