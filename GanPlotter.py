from jupyterplot import ProgressPlot
import time
import numpy as np

class GanPlotter:
    def __init__(self,moving_average_size) -> None:
        self.moving_average_size = moving_average_size
        self.progress_plot = ProgressPlot(plot_names = ['D Loss','D acc','G Loss','G acc', 'Epoch Duration'],line_names=["value", "MA"])
        self.d_loss_ma_buffer = []
        self.g_loss_ma_buffer = []
        self.d_acc_ma_buffer = []
        self.g_acc_ma_buffer = []
        self.time_ma_buffer = []
    
    def start_batch(self):
        self.epoch_start = time.time()
        self.batch_d_loss = [] 
        self.batch_g_loss = []
        self.batch_d_acc = [] 
        self.batch_g_acc = []
    
    def batch_update(self,d_loss,d_acc,g_loss,g_acc):
        self.batch_d_loss.append(d_loss)
        self.batch_g_loss.append(g_loss)
        self.batch_d_acc.append(d_acc)
        self.batch_g_acc.append(g_acc)
    
    def end_batch(self):
        self.d_loss = np.mean(self.batch_d_loss)
        self.g_loss = np.mean(self.batch_g_loss)
        self.d_acc = np.mean(self.batch_d_acc)
        self.g_acc = np.mean(self.batch_g_acc)
    
    def log_epoch(self):
        self.epoch_elapsed = time.time()-self.epoch_start
        
        self.d_loss_ma_buffer.append(self.d_loss)
        self.g_loss_ma_buffer.append(self.g_loss)
        self.d_acc_ma_buffer.append(self.d_acc)
        self.g_acc_ma_buffer.append(self.g_acc)
        self.time_ma_buffer.append(self.epoch_elapsed)
        
        d_loss_ma_buffer = self.d_loss_ma_buffer[1:] if len(self.d_loss_ma_buffer) >= self.moving_average_size else self.d_loss_ma_buffer
        g_loss_ma_buffer = self.g_loss_ma_buffer[1:] if len(self.g_loss_ma_buffer) >= self.moving_average_size else self.g_loss_ma_buffer
        d_acc_ma_buffer = self.d_acc_ma_buffer[1:] if len(self.d_acc_ma_buffer) >= self.moving_average_size else self.d_acc_ma_buffer
        g_acc_ma_buffer = self.g_acc_ma_buffer[1:] if len(self.g_acc_ma_buffer) >= self.moving_average_size else self.g_acc_ma_buffer
        time_ma_buffer = self.time_ma_buffer[1:] if len(self.time_ma_buffer) >= self.moving_average_size else self.time_ma_buffer

        d_loss_ma,g_loss_ma = np.mean(d_loss_ma_buffer),np.mean(g_loss_ma_buffer)
        d_acc_ma,g_acc_ma = np.mean(d_acc_ma_buffer), np.mean(g_acc_ma_buffer)
        time_ma = np.mean(time_ma_buffer)
        
        self.progress_plot.update([[self.d_loss,d_loss_ma],[self.d_acc,d_acc_ma],[self.g_loss,g_loss_ma],[self.g_acc,g_acc_ma],[self.epoch_elapsed,time_ma]])