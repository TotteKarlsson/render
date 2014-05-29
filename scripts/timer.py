import time

class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print "Starting", self.name
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print "    {} took {:.2f} secs".format(self.name, time.time() - self.start)
