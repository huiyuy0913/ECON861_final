import pandas
import dill as pickle

with open("machine.pickle", "rb") as f:
	machine = pickle.load(f)
	count_vectorize_transformer = pickle.load(f)
	lemmatizer = pickle.load(f)
	stopwords = pickle.load(f)
	string = pickle.load(f)
	pre_processing = pickle.load(f)




'''write your file name here'''
file_name = "sample_new"





sample_new = pandas.read_csv("exam_data/part_1/sample_new_data/" + file_name + ".csv", sep=',', skipinitialspace=True)
sample_new_transformed = count_vectorize_transformer.transform(sample_new.iloc[:,1])
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

print(sample_new)
sample_new.to_csv("exam_data/part_1/sample_new_data/" + file_name + "_with_prediction.csv", index=False, float_format='%.5f')

