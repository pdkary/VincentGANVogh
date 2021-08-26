from jupyterplot import ProgressPlot
import time
import numpy as np

class GanPlotter:
    def __init__(self,moving_average_size) -> None:
        self.moving_average_size = moving_average_size
        self.progress_plot = ProgressPlot(plot_names = ['D Loss','Disc D label','Disc G label','G Loss','Gen G label', 'Epoch Duration'],line_names=["value", "MA"])
        self.d_loss_ma_buffer = []
        self.g_loss_ma_buffer = []
        self.d_d_label_ma_buffer = []
        self.d_g_label_ma_buffer = []
        self.g_g_label_ma_buffer = []
        self.time_ma_buffer = []
    
    def start_batch(self):
        self.epoch_start = time.time()
        self.batch_d_loss = [] 
        self.batch_g_loss = []
        self.batch_d_d_avg = [] 
        self.batch_d_g_avg = []
        self.batch_g_g_avg = []
    
    def batch_update(self,d_loss,d_davg,d_gavg,g_loss,g_gavg):
        self.batch_d_loss.append(d_loss)
        self.batch_g_loss.append(g_loss)
        self.batch_d_d_avg.append(d_davg)
        self.batch_d_g_avg.append(d_gavg)
        self.batch_g_g_avg.append(g_gavg)
    
    def end_batch(self):
        self.d_loss = np.mean(self.batch_d_loss)
        self.g_loss = np.mean(self.batch_g_loss)
        self.d_davg = np.mean(self.batch_d_d_avg)
        self.d_gavg = np.mean(self.batch_d_g_avg)
        self.g_gavg = np.mean(self.batch_g_g_avg)
    
    def log_epoch(self):
        self.epoch_elapsed = time.time()-self.epoch_start
        
        self.d_loss_ma_buffer.append(self.d_loss)
        self.g_loss_ma_buffer.append(self.g_loss)
        self.d_d_label_ma_buffer.append(self.d_davg)
        self.d_g_label_ma_buffer.append(self.d_gavg)
        self.g_g_label_ma_buffer.append(self.g_gavg)
        self.time_ma_buffer.append(self.epoch_elapsed)
        
        d_loss_ma_buffer = self.d_loss_ma_buffer[1:] if len(self.d_loss_ma_buffer) >= self.moving_average_size else self.d_loss_ma_buffer
        g_loss_ma_buffer = self.g_loss_ma_buffer[1:] if len(self.g_loss_ma_buffer) >= self.moving_average_size else self.g_loss_ma_buffer
        d_d_label_ma_buffer = self.d_d_label_ma_buffer[1:] if len(self.d_d_label_ma_buffer) >= self.moving_average_size else self.d_d_label_ma_buffer
        d_g_label_ma_buffer = self.d_g_label_ma_buffer[1:] if len(self.d_g_label_ma_buffer) >= self.moving_average_size else self.d_g_label_ma_buffer
        g_g_label_ma_buffer = self.g_g_label_ma_buffer[1:] if len(self.g_g_label_ma_buffer) >= self.moving_average_size else self.g_g_label_ma_buffer
        time_ma_buffer = self.time_ma_buffer[1:] if len(self.time_ma_buffer) >= self.moving_average_size else self.time_ma_buffer

        d_loss_ma,g_loss_ma = np.mean(d_loss_ma_buffer),np.mean(g_loss_ma_buffer)
        d_d_label,d_g_label,g_g_label = np.mean(d_d_label_ma_buffer), np.mean(d_g_label_ma_buffer), np.mean(g_g_label_ma_buffer)
        time_ma = np.mean(time_ma_buffer)
        
        self.progress_plot.update([[self.d_loss,d_loss_ma],[self.d_davg,d_d_label],[self.d_gavg,d_g_label],[self.g_loss,g_loss_ma],[self.g_gavg,g_g_label],[self.epoch_elapsed,time_ma]])