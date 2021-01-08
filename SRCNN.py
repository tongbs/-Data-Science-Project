import torch
import torch.nn as nn
from main import *

def synthetic(windows):
	preceding = 5
	X, y = [], []
	for window in windows:
		data = window.copy()
		all_mean = np.mean(window)
		all_var = np.var(window)
		select_numbers = np.random.randint(1, 1440)
		select_ids = np.random.choice(1440, select_numbers, replace=False)
		lables = np.zeros(1440, dtype=np.int64)
		for _id in select_ids:
			if _id < preceding:
				continue
			local_mean = np.mean(window[_id-preceding:_id])
			data[_id] = (local_mean+all_mean)*(1+all_var)*np.random.randn()+window[_id]
			lables[_id] = 1
		X.append(data)
		y.append(lables)
	return np.array(X), np.array(y)
		

class CNN(nn.Module):
	def __init__(self):
		self.layer1 = nn.Conv1d(1440, 1440, kernel_size=1)
		self.layer2 = nn.Conv1d(1440, 2880, kernel_size=1)
		self.fc1 = nn.Linear(2880, 5760)
		self.fc2 = nn.Linear(5760, 1440)
		self.relu = nn.ReLU(inplace=True)
	def forward(self, x):
		x = x.view(x.size(0), 1440, 1)
		x = self.layer1(x)
		x = self.relu(x)
		x = self.layer2(x)
		x = x.view(x.size(0), -1)
		x = self.relu(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return torch.sigmoid(x)

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
		train_data = data_preprocess(train_data)
		X_train, y_train = sliding_window(train_data, window_size)
		
		model = SR()
		X_sr = []
		for data in X_train:
			X_sr.append(model.spectral_residual(data))
		X_syc, y_syc = synthetic(X_sr)
		
		# train
		device = torch.device('cuda:0')
		model = CNN().to(device)
		optimizer = torch.optim.SGD(model.parameters())
		Loss = torch.nn.CrossEntropyLoss()

		for epoch in range(100):
			model.train()
			X_cuda, y_cuda = torch.autograd.Variable(X_syc).to(device), torch.autograd.Variable(y_syc).to(device)
			optimizer.zero_grad()

			output = model(X_cuda)
			loss = Loss(output, y_cuda)
			loss.backward()
			optimizer.step()

		# test
		test_data = test_df[test_df['KPI ID']==kpi_name].reset_index(drop=True)
		test_data = data_preprocess(test_data)
		X_test, y_test = sliding_window(test_data, window_size)
		X_test_sr = []
		for data in X_test:
			X_test_sr.append(model.spectral_residual(data))

		model.eval()
		X_test_cuda = torch.autograd.Variable(X_test_sr).to(device)
		with torch.no_grad():
			output = model(X_test_cuda)
		y_pred = output.data.max(1)[1].cpu()



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