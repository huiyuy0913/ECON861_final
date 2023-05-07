import pandas
import dill as pickle

with open("machine.pickle", "rb") as f:
	machine = pickle.load(f)
	count_vectorize_transformer = pickle.load(f)
	lemmatizer = pickle.load(f)
	stopwords = pickle.load(f)
	string = pickle.load(f)
	pre_processing = pickle.load(f)

file_name = "dataset"

sample_new = pandas.read_csv("exam_data/part_1/training_data/" + file_name + ".csv", sep=',', skipinitialspace=True)
sample_new_transformed = count_vectorize_transformer.transform(sample_new.iloc[:,0])
print(sample_new.iloc[:,0])
prediction = machine.predict(sample_new_transformed)
prediction_prob = machine.predict_proba(sample_new_transformed)
print(prediction)
print(prediction_prob)


sample_new['prediction'] = prediction

prediction_prob_dataframe = pandas.DataFrame(prediction_prob)

prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
	prediction_prob_dataframe.columns[0]: "prediction_prob_1", 
	prediction_prob_dataframe.columns[1]: "prediction_prob_2", 
	prediction_prob_dataframe.columns[2]: "prediction_prob_3"})
# print(prediction_prob_dataframe)

sample_new = pandas.concat([sample_new, prediction_prob_dataframe], axis=1) 


sample_new['prediction'] = sample_new['prediction'].astype(int)
sample_new['prediction_prob_1'] = round(sample_new['prediction_prob_1'],5)
sample_new['prediction_prob_2'] = round(sample_new['prediction_prob_2'],5)
sample_new['prediction_prob_3'] = round(sample_new['prediction_prob_3'],5)


sample_new['misclassified'] = (sample_new['prediction'] != sample_new['stars'])*1


print(sample_new)
print(sample_new['misclassified'].value_counts())
print("\n")
print("The reviews have been misclassified are:")
print(sample_new[sample_new['misclassified']==1])


for index, row in sample_new[sample_new['misclassified']==1].iterrows():
    print("\n")
    print(row.reviewtext, row.stars, row.prediction)






