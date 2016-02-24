import os,sys
import numpy as np
import scipy.stats as stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import optimize

class pipeline(object):
    
	def __init__(self):
		self.path = 'C:\\Users\\Dober\\Desktop\\NoiseStreams\\01312016\\'
		self.option = 'IQ_Loops'
		self.select = '04'
		#self.vna_path = 'vna_sweeps'
		#self.target_path = 'target_sweeps'
		self.datapath = os.path.join(self.path, self.option,self.select)
		data_files=[f for f in sorted(os.listdir(self.datapath)) if f.endswith('.npy')]
		I = np.array([np.load(os.path.join(self.datapath,f)) for f in data_files if f.startswith('I')])
		Q = np.array([np.load(os.path.join(self.datapath,f)) for f in data_files if f.startswith('Q')])
		self.lo_freqs = np.array([np.float(f[1:-4]) for f in data_files if f.startswith('I')])
		#print f
		self.ts = I + 1j*Q
		self.nchan = len(self.ts[0])
		self.cm = plt.cm.spectral(np.linspace(0.05,0.95,self.nchan))
		self.i = self.ts.real
		self.q = self.ts.imag
		self.mag = np.abs(self.ts)
		self.phase = np.angle(self.ts)
		self.ts_off = np.load(os.path.join(self.datapath,'timestreams/I750.27.npy')) + 1j*np.load(os.path.join(self.datapath,'timestreams/Q750.27.npy'))
		self.ts_on = np.load(os.path.join(self.datapath,'timestreams/I750.57.npy')) + 1j*np.load(os.path.join(self.datapath,'timestreams/Q750.57.npy'))
		self.i_off, self.i_on = self.ts_off.real, self.ts_on.real
		self.q_off, self.q_on = self.ts_off.imag, self.ts_on.imag
		self.phase_off = np.angle(self.ts_off)	
		self.phase_on = np.angle(self.ts_on)	
		self.kid_freqs=np.load(os.path.join(self.path,'last_kid_freqs.npy'))
		self.bb_freqs=np.load(os.path.join(self.path,'last_bb_freqs.npy'))
		self.rf_freqs=np.load(os.path.join(self.path,'last_rf_freqs.npy'))
		self.delta_lo = 2.5e3
		
	def phase_scatter(self,chan):
		fig = plt.figure()
		plt.suptitle('Phase scatter, Channel = ' + str(chan))
		plot1 = plt.subplot(1,2,1)
		plot1.scatter(self.i_off[:,chan], self.q_off[:,chan], color = 'b', label = 'off res')	
		plot1.scatter(self.i_on[:,chan], self.q_on[:,chan], color = 'r', label = 'on res')
		plt.xlabel('I')
		plt.ylabel('Q')
		plot1.set_autoscale_on(True)
		off_data = sorted(self.phase_off[:,chan])
		on_data = sorted(self.phase_on[:,chan])
		fwhm_off = np.abs(np.round(2.355*np.std(self.phase_off[:,chan]),3))
		fwhm_on = np.abs(np.round(2.355*np.std(self.phase_on[:,chan]),3))
		off_fit = stats.norm.pdf(off_data, np.mean(self.phase_off[:,chan]), np.std(self.phase_off[:,chan]))
		on_fit = stats.norm.pdf(on_data, np.mean(self.phase_on[:,chan]), np.std(self.phase_on[:,chan]))
		plot2 = plt.subplot(1,2,2)
		plot2.plot(off_data - np.mean(off_data), off_fit, color = 'b', label = 'fwhm off = '+ str(fwhm_off))
		plot2.plot(on_data - np.mean(on_data), on_fit, color = 'r', label = 'fwhm on = '+ str(fwhm_on))
		plt.xlabel('rad')
		plt.ylabel('Prob. Density')
		plt.legend()
		plt.show()
		return
	
	def delta_f(self, channel):	
		i_index = [np.where(np.abs(np.diff(self.i[:,chan])) == np.max(np.abs(np.diff(self.i[:,chan]))))[0][0] for chan in range(self.nchan)]
		q_index = [np.where(np.abs(np.diff(self.q[:,chan])) == np.max(np.abs(np.diff(self.q[:,chan]))))[0][0] for chan in range(self.nchan)]
		self.di_df = np.array([(self.i[:,chan][i_index[chan] + 1] - self.i[:,chan][i_index[chan] - 1])/(2*self.delta_lo) for chan in range(self.nchan)])
		self.dq_df = np.array([(self.q[:,chan][q_index[chan] + 1] - self.q[:,chan][q_index[chan] - 1])/(2*self.delta_lo) for chan in range(self.nchan)])
		self.delta_i_on = [self.i_on[:,chan] - np.mean(self.i_on[:,chan]) for chan in range(self.nchan)] 
		self.delta_q_on = [self.q_on[:,chan] - np.mean(self.q_on[:,chan]) for chan in range(self.nchan)] 
		self.delta_i_off = [self.i_off[:,chan] - np.mean(self.i_off[:,chan]) for chan in range(self.nchan)] 
		self.delta_q_off = [self.q_off[:,chan] - np.mean(self.q_off[:,chan]) for chan in range(self.nchan)] 
		self.df_on = [ ((self.delta_i_on[chan] * self.di_df[chan]) + (self.delta_q_on[chan] * self.dq_df[chan]) / (self.di_df[chan]**2 + self.dq_df[chan]**2)) for chan in range(self.nchan)]
		self.df_off = [ ((self.delta_i_off[chan] * self.di_df[chan]) + (self.delta_q_off[chan] * self.dq_df[chan]) / (self.di_df[chan]**2 + self.dq_df[chan]**2)) for chan in range(self.nchan)]
		time = np.arange(0, len(self.i_off))/244.
		frequ = np.arange(1, len(self.i_off)+1)*244./len(self.i_off)
		plt.loglog((frequ[1:]), ((np.abs(np.fft.fft(self.df_off[channel]/self.kid_freqs[channel])))**2/len(self.i_off))[1:], label = r'$\Delta$f off', color = 'b')
		plt.loglog((frequ[1:]), ((np.abs(np.fft.fft(self.df_on[channel]/self.kid_freqs[channel])))**2/len(self.i_off))[1:], label = r'$\Delta$f on', color = 'g')
		#plt.plot(self.df_off[channel], color = 'b', label = 'off')
		#plt.plot(self.df_on[channel], color= 'g', label = 'on')
		plt.title(r'$\Delta$f, Channel = ' + str(channel))
		plt.xlabel('freq (Hz)')
		plt.ylabel(r'(ef/f$_{0}$)$^{2}$ (Hz)')
		plt.show()
		return 
p = pipeline()
