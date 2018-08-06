# coding:utf-8
import os
import logging


class LoggerHelper():
    def __init__(self):
        # create logger
        logger_name = "example"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.__setFormatter()
        self.__addStreamHanlder()

    def __setFormatter(self):
        # create formatter
        fmt = "%(asctime)-15s %(levelname)s %(filename)s line:%(lineno)d pid:%(process)d %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        self.formatter = logging.Formatter(fmt, datefmt)

    def __addStreamHanlder(self):
        # add std console handler and formatter to logger
        sh = logging.StreamHandler(stream=None)
        sh.setLevel(logging.DEBUG)
        formatter = self.formatter
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

    def addFileHanlder(self, log_path, log_name):
        # create file handler
        fh = logging.FileHandler( os.path.join( log_path, log_name)  )
        fh.setLevel(logging.INFO)
        formatter = self.formatter
        # add handler and formatter to logger
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    



