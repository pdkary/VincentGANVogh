from jupyterplot import ProgressPlot
import time
import numpy as np

class GanPlotter:
    def __init__(self,moving_average_size,labels) -> None:
        self.moving_average_size = moving_average_size
        self.labels = labels + ["Epoch Time"]
        self.progress_plot = ProgressPlot(plot_names = labels,line_names=["value", "MA"])
        self.num_labels = len(labels)
        self.plot_lists = [[] for i in range(self.num_labels)]
        self.plot_ma_lists = [[] for i in range(self.num_labels)]
        self.batch_lists = [[] for i in range(self.num_labels)]
    
    def start_epoch(self):
        self.epoch_start = time.time()
        self.batch_lists = [[] for i in range(self.num_labels)]
    
    def batch_update(self,update_arr):
        assert len(update_arr) == len(self.labels)-1, "values must have same length as labels"
        for i,val in enumerate(update_arr):
            self.batch_lists[i].append(val)
    
    def log_epoch(self):
        epoch_time = time.time()-self.epoch_start
        self.plot_lists[-1].append(epoch_time)
        output = []
        for i in range(self.num_labels):
            batch_mean = np.mean(self.batch_lists[i])
            self.plot_lists[i].append(batch_mean)
            self.plot_ma_lists[i].append(batch_mean)
            self.plot_ma_lists[i] = self.plot_ma_lists[i][1:] if len(self.plot_ma_lists[i]) >= self.moving_average_size else self.plot_ma_lists[i]
            batch_ma = np.mean(self.plot_ma_lists[i])
            output.append([batch_mean,batch_ma])
        self.progress_plot.update(output)