
import pickle
import pandas
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy
import glob

import dill as pickle




machine = pickle.load(open('machine_part2_cnn_image.pickle', 'rb'))


new_data = ImageDataGenerator(rescale = 1/255).flow_from_directory('exam_data/part_2/sample_new_data/',
                                             target_size=(30,30), batch_size=1, shuffle=False)

new_data.reset()


'''write your folder name here'''
folder_name = "sample_profile_pictures"
'''write your file name here'''
file_name = "sample_new"


new_data_length = len([i for i in glob.glob("exam_data/part_2/sample_new_data/" + folder_name + "/*.jpg")])

prediction = numpy.argmax(machine.predict_generator(new_data,steps = new_data_length), axis=1)

results = [[new_data.filenames[i], prediction[i]]for i in range(new_data_length)]
results = pandas.DataFrame(results, columns=['profile_picture', 'profile_prediction'])
results['profile_picture'] = results['profile_picture'].str.replace("" + folder_name + "/", '')

sample_new = pandas.read_csv("exam_data/part_2/sample_new_data/" + file_name + ".csv", skipinitialspace=True)
# print(sample_new)
# print(sample_new.columns)
# print(results.columns)
# sample_new.columns = [sample_new.columns[0], 'profile_picture']
# sample_new['profile_picture'] = sample_new['profile_picture'].str.replace(' ', '')
# print(results)
results = sample_new.merge(results, on=['profile_picture'], suffixes=['_new','_results'], how='left')



label_map = {0: 'building', 1: 'dog', 2: 'face'}
results['profile'] = results['profile_prediction'].map(label_map)

print(results)




with open("machine_part2.pickle", "rb") as f:
    machine = pickle.load(f)
    count_vectorize_transformer = pickle.load(f)
    lemmatizer = pickle.load(f)
    stopwords = pickle.load(f)
    string = pickle.load(f)
    pre_processing = pickle.load(f)



results_transformed = count_vectorize_transformer.transform(results.profile)
prediction = machine.predict(results_transformed)
prediction_prob = machine.predict_proba(results_transformed)
# print(prediction)
# print(prediction_prob)


results['star_prediction'] = prediction

prediction_prob_dataframe = pandas.DataFrame(prediction_prob)

prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
    prediction_prob_dataframe.columns[0]: "star_prediction_prob_1", 
    prediction_prob_dataframe.columns[1]: "star_prediction_prob_2", 
    prediction_prob_dataframe.columns[2]: "star_prediction_prob_3"})
# print(prediction_prob_dataframe)

results = pandas.concat([results, prediction_prob_dataframe], axis=1) 


results['star_prediction'] = results['star_prediction'].astype(int)
results['star_prediction_prob_1'] = round(results['star_prediction_prob_1'],5)
results['star_prediction_prob_2'] = round(results['star_prediction_prob_2'],5)
results['star_prediction_prob_3'] = round(results['star_prediction_prob_3'],5)

print(results)
results.to_csv("exam_data/part_2/sample_new_data/" + file_name + "_data_prediction.csv", index=False, float_format='%.5f')




