# from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import numpy
import pandas

def run_kfold(data_1,data_2, target, machine_1, machine_2, n, prediction_probability=False):

	# print("run kfold")
	kfold_object = KFold(n_splits=n)
	kfold_object.get_n_splits(data_1)

	# print(kfold_object)


	
	all_return_values = []
	i = 0
	for train_index, test_index in kfold_object.split(data_1):
		i=i+1
		# print("Round:", str(i))
		# print("Training index:")
		# print(train_index)
		# print("Testing index")
		# print(test_index)
		
		data_train_1 = data_1[train_index]
		target_train = target[train_index]
		data_test_1 = data_1[test_index]
		target_test = target[test_index]
		# print(data_train_1)

		machine_1.fit(data_train_1, target_train)
		# prediction_1 = machine_1.predict(data_test_1)


		data_train_2 = data_2[train_index]
		data_test_2 = data_2[test_index]
		# print(data_train_1 != data_train_2)
		machine_2.fit(data_train_2, target_train)
		# prediction_2 = machine_2.predict(data_test_2)
		# print(machine_1 != machine_2)


		return_value = []
		# if (use_r2 == True):	
		# 	r2_1 = metrics.r2_score(target_test, prediction_1)
		# 	# print("R square score: ",r2)
		# 	return_value.append(r2_1)
		if (prediction_probability == True):
			prediction_prob_1 = machine_1.predict_proba(data_test_1)
			prediction_prob_2 = machine_2.predict_proba(data_test_2)
			# print(prediction_prob_1)
			# print(prediction_prob_2)
			prediction_prob_dataframe_1 = pandas.DataFrame(prediction_prob_1)
			prediction_prob_dataframe_1 = prediction_prob_dataframe_1.rename(columns={
				prediction_prob_dataframe_1.columns[0]: "prediction_prob_1", 
				prediction_prob_dataframe_1.columns[1]: "prediction_prob_2", 
				prediction_prob_dataframe_1.columns[2]: "prediction_prob_3"})
			# print(prediction_prob_dataframe_1)
			prediction_prob_dataframe_2 = pandas.DataFrame(prediction_prob_2)
			prediction_prob_dataframe_2 = prediction_prob_dataframe_2.rename(columns={
				prediction_prob_dataframe_2.columns[0]: "prediction_prob_1", 
				prediction_prob_dataframe_2.columns[1]: "prediction_prob_2", 
				prediction_prob_dataframe_2.columns[2]: "prediction_prob_3"})
			# print(prediction_prob_dataframe_1)
			# print(prediction_prob_dataframe_2)

			prediction_prob_dataframe_1['prediction_prob_1'] = (prediction_prob_dataframe_1['prediction_prob_1'] + prediction_prob_dataframe_2['prediction_prob_1'])/2
			prediction_prob_dataframe_1['prediction_prob_2'] = (prediction_prob_dataframe_1['prediction_prob_2'] + prediction_prob_dataframe_2['prediction_prob_2'])/2
			prediction_prob_dataframe_1['prediction_prob_3'] = (prediction_prob_dataframe_1['prediction_prob_3'] + prediction_prob_dataframe_2['prediction_prob_3'])/2
			# print(prediction_prob_dataframe_1)
			probabilities = prediction_prob_dataframe_1.values
			prediction = numpy.argmax(probabilities, axis=1)+1
			accuracy = metrics.accuracy_score(target_test, prediction)
			return_value.append(accuracy)

			train_prediction_prob_1 = machine_1.predict_proba(data_train_1)
			train_prediction_prob_2 = machine_2.predict_proba(data_train_2)

			train_prediction_prob_dataframe_1 = pandas.DataFrame(train_prediction_prob_1)
			train_prediction_prob_dataframe_1 = train_prediction_prob_dataframe_1.rename(columns={
				train_prediction_prob_dataframe_1.columns[0]: "prediction_prob_1", 
				train_prediction_prob_dataframe_1.columns[1]: "prediction_prob_2", 
				train_prediction_prob_dataframe_1.columns[2]: "prediction_prob_3"})

			train_prediction_prob_dataframe_2 = pandas.DataFrame(train_prediction_prob_2)
			train_prediction_prob_dataframe_2 = train_prediction_prob_dataframe_2.rename(columns={
				train_prediction_prob_dataframe_2.columns[0]: "prediction_prob_1", 
				train_prediction_prob_dataframe_2.columns[1]: "prediction_prob_2", 
				train_prediction_prob_dataframe_2.columns[2]: "prediction_prob_3"})
			# print(train_prediction_prob_dataframe_1)
			# print(train_prediction_prob_dataframe_2)
			train_prediction_prob_dataframe_1['prediction_prob_1'] = (train_prediction_prob_dataframe_1['prediction_prob_1'] + train_prediction_prob_dataframe_2['prediction_prob_1'])/2
			train_prediction_prob_dataframe_1['prediction_prob_2'] = (train_prediction_prob_dataframe_1['prediction_prob_2'] + train_prediction_prob_dataframe_2['prediction_prob_2'])/2
			train_prediction_prob_dataframe_1['prediction_prob_3'] = (train_prediction_prob_dataframe_1['prediction_prob_3'] + train_prediction_prob_dataframe_2['prediction_prob_3'])/2

			probabilities = train_prediction_prob_dataframe_1.values
			train_prediction = numpy.argmax(probabilities, axis=1)+1
			train_accuracy = metrics.accuracy_score(target_train, train_prediction)
			return_value.append(train_accuracy)			

		# if (use_confusion == True):
		# 	confusion_1 = metrics.confusion_matrix(target_test, prediction_1)
		# 	confusion_2 = metrics.confusion_matrix(target_test, prediction_2)

		# 	# print("Confusion matrix:\n ", confusion)
		# 	return_value.append(confusion_1)


		# print("\n\n") # means two new lines
		all_return_values.append(return_value)
	return all_return_values


if __name__ == '__main__':
    run_kfold()