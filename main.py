import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from SR_test import SR

def data_preprocess(data):
	# missing -> impute previous value
	max_time = data['timestamp'].max()
	min_time = data['timestamp'].min()
	all_time = [t for t in range(min_time, max_time+60, 60)]
	missing_time = sorted(set(all_time)-set(data['timestamp'].values))
	for t in missing_time:
		prev_col = data.loc[data.timestamp==t-60].copy()
		prev_col['timestamp'] = t
		data = data.append(prev_col, ignore_index=True)
	data.sort_values(by='timestamp', ignore_index=True, inplace=True)
	return data

def sliding_window(data, window_size):
	X = []
	for i in range(len(data)-window_size):
		time_series = data[i:i+window_size]['value'].to_list()
		X.append(time_series)
	y = data[window_size:]['label'].to_list()
	return np.array(X), np.array(y)

def feature_extraction(datas):
	X = []
	model = SR()
	for data in datas:
		#all_avg = data.mean()
		#all_std = data.std()
		#local_avg = data[-20:].mean()
		#local_std = data[-20:].std()
		#X.append([(local_avg-all_avg)/all_avg, (local_std-all_std)/all_std, (data[-1]-all_avg)/all_std, local_avg, local_std])
		"""if data.std() == 0:
			X.append([np.median(data), data.mean(), data.max(), data.min(), data.std(), data[-1], data[-1]-data.mean()])
		else:
			X.append([np.median(data), data.mean(), data.max(), data.min(), data.std(), data[-1], (data[-1]-data.mean())/data.std()])"""
		X.append(model.feature(data))
	return np.array(X)

# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict

if __name__ == '__main__':

	train_df = pd.read_csv('./Dataset/phase2_train.csv')
	test_df = pd.read_hdf('./Dataset/phase2_ground_truth.hdf')
	test_df['KPI ID'] = test_df['KPI ID'].astype(str) # UUID to str

	kpi_names = train_df['KPI ID'].values
	kpi_names = np.unique(kpi_names)

	window_size = 1440
	delay = 7

	y_true_list, y_pred_list = [], []
	for kpi_num, kpi_name in enumerate(kpi_names):
		print(f'===========KPI#{kpi_num+1}: {kpi_name}===========')

		train_data = train_df[train_df['KPI ID']==kpi_name].reset_index(drop=True)
		#train_data = train_df[train_df['KPI ID']==kpi_name].reset_index(drop=True)
		train_data = data_preprocess(train_data)
		X_train, y_train = sliding_window(train_data, window_size)
		X_train = feature_extraction(X_train)

		#test_data = test_df[test_df['KPI ID']==kpi_name][0:10000].reset_index(drop=True)
		test_data = test_df[test_df['KPI ID']==kpi_name].reset_index(drop=True)
		test_data = data_preprocess(test_data)
		X_test, y_test = sliding_window(test_data, window_size)
		X_test = feature_extraction(X_test)

		#model = svm.OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
		#model = IsolationForest()
		#model.fit(X_train)
		#y_pred = model.predict(X_test)

		#y_pred[y_pred==1]=0
		#y_pred[y_pred==-1]=1


		model = RandomForestClassifier()
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

		"""model = SR()
		y_pred = []
		for x in X_test:
			y_pred.append(model.detection(x))"""

		y_pred2 = get_range_proba(y_pred, y_test, delay)

		#fscore = f1_score(np.concatenate(y_test), np.concatenate(y_pred))
		fscore = f1_score(y_test, y_pred)
		print(fscore)
		print(classification_report(y_test, y_pred))

		fscore = f1_score(y_test, y_pred2)
		print(fscore)
		print(classification_report(y_test, y_pred2))

		np.savez(f'./Result/Result_{kpi_name}.npz', y_test=y_test, y_pred=y_pred, y_pred2=y_pred2)
		y_true_list.append(y_test)
		y_pred_list.append(y_pred2)
		
	
	print('=============')
	fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
	print(fscore)
	print(classification_report(np.concatenate(y_true_list), np.concatenate(y_pred_list)))


"""df = pd.read_csv('./phase2_train.csv')
print(df.shape)
print(df['KPI ID'].value_counts())
print(df[df['label']==1].shape)


data = df[df['KPI ID']=='da10a69f-d836-3baa-ad40-3e548ecf1fbd'][0:25000]

plt.figure(figsize=(20,5))
data['value'].plot()

start, end = [], []
prev = -2
for i in data[data['label']==1].index.to_list():
    if prev+1 != i:
        end.append(prev)
        start.append(i)
    prev = i
end = end[1:]+[prev]
#print(start)
#print(end)
for i in range(len(start)):
    plt.axvspan(start[i], end[i], facecolor='r', alpha=0.5)
plt.savefig('result.png')"""