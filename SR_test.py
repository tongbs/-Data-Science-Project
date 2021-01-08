import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SR:
	def detection(self, time_series, neighbors=21, threshold=3):
		ext_time_series = self.extend_time_series(time_series)
		SR_time_series = self.spectral_residual(ext_time_series)
		n = len(time_series)-1
		#local_average = SR_time_series[n-neighbors:n].mean()
		local_average = SR_time_series[:n].mean()
		local_std = SR_time_series[:n].std()
		#if (SR_time_series[n]-local_average)/local_average >= threshold:
		if (SR_time_series[n]-local_average)/local_std >= threshold:
			return 1
		else:
			return 0
	def feature(self, time_series, neighbors=21):
		ext_time_series = self.extend_time_series(time_series)
		SR_time_series = self.spectral_residual(ext_time_series)
		n = len(time_series)-1
		local_average = SR_time_series[n-neighbors:n].mean()
		local_std = SR_time_series[n-neighbors:n].std()
		all_average = SR_time_series[:n].mean()
		all_std = SR_time_series[:n].std()
		if all_std == 0:
			return np.array([local_average, local_std, SR_time_series[n], (SR_time_series[n]-all_average), SR_time_series[n]])
		else:
			return np.array([local_average, local_std, SR_time_series[n]/local_average, (SR_time_series[n]-all_average)/all_std, SR_time_series[n]])
	def spectral_residual(self, time_series):
		fft = np.fft.fft(time_series)
		# numerical imprecision
		val = np.sqrt(fft.real**2+fft.imag**2)
		error_index = np.where(val<=1e-8)[0]
		val[error_index] = 1e-8

		log_val = np.log(val)
		log_val[error_index] = 0

		avg_sp = self.convoluting(log_val)
		res = np.exp(log_val-avg_sp)

		fft.real = fft.real * res / val
		fft.imag = fft.imag * res / val
		fft.real[error_index] = 0
		fft.imag[error_index] = 0

		wave = np.fft.ifft(fft)
		result = np.sqrt(wave.real**2+wave.imag**2)
		return result
	def convoluting(self, time_series, q=3):
		mat = np.cumsum(time_series)
		mat[q:] = mat[q:] - mat[:-q]
		mat[q:] = mat[q:]/q
		for i in range(1, q):
			mat[i] /= (i+1)
		return mat
	def extend_time_series(self, time_series, preceding=5, extend=5):
		selected_time_series = time_series[-preceding-1:]
		gradient = np.mean([(selected_time_series[-1]-selected_time_series[-1-i])/i for i in range(1, preceding+1)])
		estimated_points = selected_time_series[1] + preceding*gradient
		return np.concatenate((time_series, [estimated_points]*extend))
	
if __name__ == '__main__':

	data = np.load('data.npy')

	model = SR()
	print(model.detection(data))
	

	new_data = model.extend_time_series(data)
	print(new_data.shape, data.shape)
	result = model.spectral_residual(new_data)
	plt.plot([i for i in range(len(result))], result)
	print(result)


	plt.plot([i for i in range(len(data))], data)

	plt.legend(['SR', 'Time Series'])
	plt.savefig('SR.png')