import codecs
import datetime
import time



class Logger():
    def __init__(self, location):
        self.f = codecs.open(location, 'w', encoding='utf-8')

    def log(self, s, show_time=True):
        if show_time:
            curtime = time.time()
            ts = datetime.datetime.fromtimestamp(curtime).strftime('%Y-%m-%d %H:%M:%S')
            self.f.write(ts + ': ' + s + '\n')
        else:
            self.f.write(s + '\n')


    def close(self):
        self.f.close()
