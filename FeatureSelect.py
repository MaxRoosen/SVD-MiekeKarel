# load and summarize the dataset
import numpy as np
import pandas as pd
import shutil
import datetime
import os
import re
import seaborn as sn
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
	data = read_csv(filename, header=0, delimiter=';')
	# remove unwanted columns
	if 'ppnnr' in data:
		data = data.drop(['ppnnr'], axis='columns')
	# if 'DatInfo' in data:
	# 	data = data.drop(['DatInfo'],axis='columns')
	# if 'Datbloed' in data:
	# 	data = data.drop(['Datbloed'],axis='columns')
	# if 'DatMRI' in data:
	# 	data = data.drop(['DatMRI'],axis='columns')
	# if 'Symp_Inf_code' in data:
	# 	data = data.drop(['Symp_Inf_code'],axis='columns')	
	# if 'Symp_Inf_aanw' in data:
	# 	data = data.drop(['Symp_Inf_aanw'], axis='columns')
	# if 'PF4' in data:
	# 	data = data.drop(['PF4'], axis='columns')
	# if 'MPO' in data:
	# 	data = data.drop(['MPO'], axis='columns')
	# if 'fractalkine' in data:
	# 	data = data.drop(['fractalkine'], axis='columns')
	# if 'B-TG' in data:
	# 	data = data.drop(['B-TG'], axis='columns')
	# Recode Group 
	data['Group'].replace({1:0, 2:0, 3:1}, inplace=True)
	# Remove null values
	data = data.dropna()
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	#X = dataset[:, :-1]
	X = data[data.columns[~data.columns.isin(['Group'])]]
	y = data['Group']
	# format all fields as string
	#X = X.astype(str)
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	trans = MinMaxScaler()
	#trans.fit(X_train)
	X_train_enc = trans.fit_transform(X_train)
	X_test_enc = trans.fit_transform(X_test)
	return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# feature selection
def select_features(X_train, y_train, X_test, num_feat):
	fs = SelectKBest(score_func=f_classif, k=num_feat)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# sensitivity & specificity 
def calculate_sensitivity_specificity(y_test, y_pred_test):
    # Note: More parameters are defined than necessary. 
    # This would allow return of other measures other than sensitivity and specificity
    
    # Get true/false for whether a breach actually occurred
    actual_pos = y_test == 1
    actual_neg = y_test == 0
    
    # Get true and false test (true test match actual, false tests differ from actual)
    true_pos = (y_pred_test == 1) & (actual_pos)
    false_pos = (y_pred_test == 1) & (actual_neg)
    true_neg = (y_pred_test == 0) & (actual_neg)
    false_neg = (y_pred_test == 0) & (actual_pos)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_test == y_test)
    
    # Calculate sensitivity and specificity
    sensitivity = np.sum(true_pos) / np.sum(actual_pos)
    specificity = np.sum(true_neg) / np.sum(actual_neg)
    
    return sensitivity, specificity, accuracy

# wrapper script
def run_full_script(num_feat, file_name):
	# load the dataset
	X, y = load_dataset(file_name)
	# split into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
	# prepare output data
	y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
	# prepare input data
	#x_train_enc, x_test_enc = prepare_inputs(X_train, X_test)

	# feature selection
	X_train_fs, X_test_fs, fs = select_features(X_train, y_train_enc, X_test, num_feat)

	images = np.array(fs.pvalues_)
	label = np.array(X.columns.values)
	dataset = pd.DataFrame({'Feature': label, 'PValue': list(images)}, columns=['Feature', 'PValue'])

	# Prepare data for result plotting

	dataset = dataset.sort_values('PValue',ascending = True).head(num_feat)
	dataset = dataset.reset_index()

	# what are scores for the features save to file
	filedate = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	result_file = "Results_"+filedate+"_"+file_name.replace('.csv','')+'_feat_'+str(num_feat)
	file = open(result_file +".txt", "w")
	file.write('Input_File: ' + file_name + '\n') 
	for i in range(len(dataset.index)):
		#print('Feature %s: %f' % (dataset.at[i,'Feature'], dataset.at[i,'PValue']))
		file.write('P-Value %s: %f \n' % (dataset.at[i,'Feature'], dataset.at[i,'PValue'])) 

	# plot the scores
	plt.rcParams.update({'font.size': 6})
	n = dataset.Feature
	s = dataset.PValue
	line = plt.bar(n,s, color='blue')
	plt.xlabel('Feature')
	plt.xticks(rotation=90)
	plt.ylabel("Value")

	for i in range(len(s)):
		plt.annotate(str("{:.5f}".format(s[i])), xy=(n[i],s[i]), ha='center', va='bottom', rotation=90)

	# Train model
	model = LogisticRegression(solver='liblinear', max_iter=10000)
	model.fit(X_train_fs, y_train_enc)
	# evaluate the model
	yhat = model.predict(X_test_fs)
	# evaluate predictions
	sensitivity, specificity, accuracy2 = calculate_sensitivity_specificity(y_test_enc, yhat)
	accuracy = accuracy_score(y_test_enc, yhat)
	accuracy = str('%.3f' % (accuracy*100))
	# feature coefficients
	coefficients = pd.concat([pd.DataFrame(dataset.Feature),pd.DataFrame(np.transpose(model.coef_))], axis = 1)
	for i in range(len(coefficients.index)):
	 	#print('Feature: %s, coefficient: %f' % (coefficients.at[i,'Feature'], coefficients.at[i, 0]))
		 file.write('Coefficient %s: %f \n' % (coefficients.at[i,'Feature'], coefficients.at[i, 0]))
	
	#print('Accuracy: '+accuracy)
	file.write('Accuracy: '+accuracy+'\n')
	file.write('Sensitivity: '+str('%.3f' % (sensitivity*100))+'\n')
	file.write('Specificity: '+str('%.3f' % (specificity*100))+'\n')
	file.write('Accuracy_manual: '+str('%.3f' % (accuracy2*100))+'\n')
	lr_auc = roc_auc_score(y_test_enc, yhat)
	lr_auc = str('%.3f' % (lr_auc*100))
	file.write('AUC: '+lr_auc+'\n')
	file.close()
	os.rename(result_file+'.txt', result_file+'_acc_'+accuracy+'_auc_'+lr_auc+".txt")
	plt.savefig(result_file+'_box.png', bbox_inches='tight')
	plt.close('all')

	plot_roc_curve(model, X_test_fs, y_test_enc)  
	plt.savefig(result_file+'_roc.png')
	plt.close('all') 

input_file = 'Dataset_7.1.csv'
i = 1
# run_full_script(3, input_file)
while i <= 16:
	run_full_script(i, input_file)
	i += 1

all_files = os.listdir()
target = input_file.replace('.csv','')+'_Results'
if not os.path.exists(target):
    os.makedirs(target)

for file in all_files:
	if file.startswith('Results_'):
		shutil.move(file, target)


        