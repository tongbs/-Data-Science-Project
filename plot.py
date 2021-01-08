import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

train_df = pd.read_csv('./Dataset/phase2_train.csv')
test_df = pd.read_hdf('./Dataset/phase2_ground_truth.hdf')
test_df['KPI ID'] = test_df['KPI ID'].astype(str)

print(train_df.shape)
print(test_df.shape)
print(train_df[train_df['label']==1].shape)
print(test_df[test_df['label']==1].shape)


select_kpi = 'ba5f3328-9f3f-3ff5-a683-84437d16d554'
#select_kpi = '57051487-3a40-3828-9084-a12f7f23ee38'
data = test_df[test_df['KPI ID']==select_kpi][:11440]
#data = test_df[test_df['KPI ID']==select_kpi][-35000:]


data = data_preprocess(data)
data = data[1440:11440].reset_index(drop=True)
#data = data[-35000:-15000].reset_index(drop=True)


# ground truth
plt.figure(tight_layout=True,figsize=(20,5))
data['value'].plot()

start, end = [], []
prev = -2
for i in data[data['label']==1].index.to_list():
	if prev+1 != i:
		end.append(prev)
		start.append(i)
	prev = i
end = end[1:]+[prev]
print(start, end)
for i in range(len(start)):
    plt.axvspan(start[i]-1.5, end[i]+1.5, facecolor='r', alpha=0.5)

plt.savefig(f'result_{select_kpi}_groundtruth.png')
plt.clf()

# predict
plt.figure(tight_layout=True,figsize=(20,5))
data['value'].plot()

start, end = [], []
prev = -2
lab = np.load(f'./Result_std3/Result_{select_kpi}.npz')['y_pred2'][0:10000]
#lab = np.load(f'./Result_std3/Result_{select_kpi}.npz')['y_pred2'][-35000:-15000]
print(lab)
for i in np.where(lab==1)[0]:
	if prev+1 != i:
		end.append(prev)
		start.append(i)
	prev = i
end = end[1:]+[prev]
print(start, end)
for i in range(len(start)):
    plt.axvspan(start[i]-1.5, end[i]+1.5, facecolor='C1', alpha=0.5)

plt.savefig(f'result_{select_kpi}_predict.png')

"""
df = pd.read_csv('./phase2_train.csv')
kpi_id = df.groupby(df['KPI ID'])
data = kpi_id.get_group("da10a69f-d836-3baa-ad40-3e548ecf1fbd")
#print(data)
train_data = data[:80787]
test_data = data[80787:107718]
sliding_win = 7
print(len(train_data))
oneclass = svm.OneClassSVM(kernel='rbf', gamma=0.001, nu=0.5)
train_split_data = list()
for i in range(len(train_data)-sliding_win):
	if not np.any(train_data[i:i+sliding_win]['label']):
		tmp = train_data[i:i+sliding_win]['value']
		train_split_data.append([tmp.min(), tmp.max(), tmp.mean(), tmp.std()])

test_split_data = list()
test_label = list()
for i in range(len(test_data)-sliding_win):
	tmp = test_data[i:i+sliding_win]['value']
	test_split_data.append([tmp.min(), tmp.max(), tmp.mean(), tmp.std()])
	test_label.append(np.any(test_data[i:i+sliding_win]['label']))
#print(test_label)

test_label = [-1 if obj else 1 for obj in test_label]

	
oneclass.fit(train_split_data)
test_res = oneclass.predict(test_split_data)
print(test_res)
print(confusion_matrix(test_label, test_res).ravel())
tn, fp, fn, tp = confusion_matrix(test_label, test_res).ravel()
print((tn+tp)/(tn+fp+fn+tp))"""



#train_nor_data = train_data[train_data['label'] == 0]
#train_abnor_data = train_data[train_data['label'] == 1]