# ECON861_final
## Part 1
### [part1_private_program.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part1_private_program.py)
In this code, I train the Naive Bayesian model to use the review texts to predict the programmers' stars. The machine is saved as [machine.pickle](https://github.com/huiyuy0913/ECON861_final/blob/main/machine.pickle).

### [part1_compare.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part1_compare.py)
I use this code to compare the preformance (KFold) of different models, including the Naive Bayesian model, the Logistic Regression Model, and the Random Forest model. Finally, I find the Naive Bayesian model is the best. 
The KFold template I use is [kfold_template.py](https://github.com/huiyuy0913/ECON861_final/blob/main/kfold_template.py).

### [part1_misclassified.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part1_misclassified.py)
I use this code to identify the misclassified reviews from the training dataset. The reason why there will be some misclassified reviews are listed in [code_introduction_and_explanation.pdf](https://github.com/huiyuy0913/ECON861_final/blob/main/code_introduction_and_explanation.pdf).

### [part1_client_program.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part1_client_program.py)
This is the code for the boss to predict the quality of programmers. The short instruction is in [code_introduction_and_explanation.pdf](https://github.com/huiyuy0913/ECON861_final/blob/main/code_introduction_and_explanation.pdf).

## Part 2
### [part2_profile_predict.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part2_profile_predict.py)
You can use this code to get the machine to use the profile picture types to predict the programmers' stars. The machine is saved as [machine_part2.pickle](https://github.com/huiyuy0913/ECON861_final/blob/main/machine_part2.pickle).
In this code, I tried different models, but based on their performance, I choose the Naive bayesian model. The explanation can be found in [code_introduction_and_explanation.pdf](https://github.com/huiyuy0913/ECON861_final/blob/main/code_introduction_and_explanation.pdf).
The KFold template I use is [kfold_template.py](https://github.com/huiyuy0913/ECON861_final/blob/main/kfold_template.py).

### [part2_classify_profile_picture_type.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part2_classify_profile_picture_type.py)
In this code, I train a CNN model to classify the profile picture type from the profile picture. I tried different models and finally choose the one saved in [machine_part2_cnn_image.pickle](https://github.com/huiyuy0913/ECON861_final/blob/main/machine_part2_cnn_image.pickle).
The reason can be found in [code_introduction_and_explanation.pdf](https://github.com/huiyuy0913/ECON861_final/blob/main/code_introduction_and_explanation.pdf).

### [part2_profile_predict.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part2_profile_predict.py)
This is the code for the boss to predict the quality of programmers. The short instruction is in [code_introduction_and_explanation.pdf](https://github.com/huiyuy0913/ECON861_final/blob/main/code_introduction_and_explanation.pdf).

## Part 3
### [part3_private_program.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part3_private_program.py)
In this code, I train the Naive Bayesian model to combine the review texts and the profile picture types to one column to predict the programmers' stars. The machine is saved as [machine_part3.pickle](https://github.com/huiyuy0913/ECON861_final/blob/main/machine_part3.pickle).
I tried different methods to combine the review texts and the profile picture types. The explanation can be found in [code_introduction_and_explanation.pdf](https://github.com/huiyuy0913/ECON861_final/blob/main/code_introduction_and_explanation.pdf).
I use two types of KFold templates. One is [kfold_template.py](https://github.com/huiyuy0913/ECON861_final/blob/main/kfold_template.py), the other one is [kfold_template_part3.py](https://github.com/huiyuy0913/ECON861_final/blob/main/kfold_template_part3.py).
I use [kfold_template_part3.py](https://github.com/huiyuy0913/ECON861_final/blob/main/kfold_template_part3.py) in the third method to average the separate prediction probability of the review text and the profiles. Then, use the average prediction probability to get the prediction results.

### [part3_classify_profile_picture_type.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part3_classify_profile_picture_type.py)
I use the same CNN model in part 2 to classify the profile picture type from the profile picture. The machine is saved in [machine_part3_cnn_image.pickle](https://github.com/huiyuy0913/ECON861_final/blob/main/machine_part3_cnn_image.pickle).

### [part3_client_program.py](https://github.com/huiyuy0913/ECON861_final/blob/main/part3_client_program.py)
This is the code for the boss to predict the quality of programmers. The short instruction is in [code_introduction_and_explanation.pdf](https://github.com/huiyuy0913/ECON861_final/blob/main/code_introduction_and_explanation.pdf).
