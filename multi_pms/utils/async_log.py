#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import time
import os
import re
import colorama
from colorama import Fore, Style
from logging.handlers import TimedRotatingFileHandler

colorama.init()


class AsyncLog(object):
    def __init__(self, logger=None, file_name=None):
        # Create a logger.
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        self.log_time = time.strftime("%Y_%m_%d")
        if not file_name:
            self.log_path = "./logger_file/scheduling.log"
        else:
            self.log_path = "./logger_file/" + file_name

        # logging.getLogger() returns the same object for the same name; repeated AsyncLog construction
        # must not add handlers again, or each message will be emitted multiple times.
        if self.logger.handlers:
            return

        # Create a file logging handler.
        if not os.path.exists("./logger_file/"):
            os.makedirs("./logger_file/")

        fh = TimedRotatingFileHandler(filename=self.log_path, when="MIDNIGHT", interval=1, backupCount=3)
        fh.suffix = "%Y-%m-%d.log"
        fh.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Set the handler output format.
        formatter = logging.Formatter(
            "[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, message):
        self.logger.info(message)

    def _log_color(self, message, color, bright=False):
        style = Style.BRIGHT if bright else ""
        self.logger.info(color + style + message + Style.RESET_ALL)

    def info_green(self, message):
        self._log_color(message, Fore.GREEN)

    def info_yellow(self, message):
        self._log_color(message, Fore.YELLOW)
    
    def info_cyan(self, message):
        self._log_color(message, Fore.CYAN)

    def info_blue(self, message):
        self._log_color(message, Fore.BLUE)

    def info_orange(self, message):
        self._log_color(message, "\033[38;5;172m")

    def info_green_bright(self, message):
        self._log_color(message, Fore.GREEN, bright=True)

    def info_yellow_bright(self, message):
        self._log_color(message, Fore.YELLOW, bright=True)

    def info_red_bright(self, message):
        self._log_color(message, Fore.RED, bright=True)

    def warning(self, message):
        self.logger.warning(message)
    
    def debug(self, message):
        self.logger.debug(message)
