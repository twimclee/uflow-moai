import time
import csv
from datetime import datetime, timedelta


class TimeManager:
    def __init__(self):
        self.since = time.time()

    def start(self):
        self.since = time.time()

    def get_time_left(self, epoch, epochs, start_epoch=0):
	
        time_now = time.time()
        elapsed = time_now - self.since
        epochs_done = (epoch - start_epoch + 1)
        avg_epoch_time = elapsed / epochs_done if epochs_done > 0 else 0
        epochs_left = (epochs - 1) - epoch
        eta_sec = max(avg_epoch_time * epochs_left, 0) # avg_epoch_time * epochs_left

        return str(timedelta(seconds=int(eta_sec)))


class CSVManager:
    def __init__(self, path, header):

        self.result_file = open(path, mode='a', newline='', encoding='utf-8')
        self.result_csv = csv.writer(self.result_file)
        self.result_csv.writerow(header)

    def writerow(self, data):
        self.result_csv.writerow(data)
        self.result_file.flush()

    def close(self):
        self.result_file.close()