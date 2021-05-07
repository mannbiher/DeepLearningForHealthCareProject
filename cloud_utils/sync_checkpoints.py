"""Saves model checkpoint to S3

This module saves the checkpoint data to S3 bucket when a new file is created.
FLANNEL has in-built checkpoint function that saves data that acts as restore point
after each epoch to resume from in case of any crash of machine
"""

import time
import logging
import os

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import boto3

S3_CLIENT = boto3.client('s3')
S3_BUCKET = 'alchemists-uiuc-dlh-spring2021-us-east-2'
WATCH_DIR = os.path.expanduser('~/DeepLearningForHealthCareProject/src/explore_version_03/')
logger = logging.getLogger(__name__)
FLAG_DIR = os.path.expanduser('~/DeepLearningForHealthCareProject/src/flag_dir/')


def delete_file(filename):
    try:
        os.remove(filename)
    except OSError:
        logger.exception('Delete failed')


class Handler(FileSystemEventHandler):
    def upload_file(self, event):
        if event.is_directory:
            return
        path = event.src_path
        s3_key = path.replace(WATCH_DIR, '')
        S3_CLIENT.upload_file(path,S3_BUCKET, 'flannel/' + s3_key)
        if path.endswith('checkpoint.pth.tar'):
            delete_file(path)

    def sync_directory(self):
        try:
            os.system("aws s3 sync {} {}".format(WATCH_DIR, "s3://" + S3_BUCKET + "/flannel/"))
            os.remove(FLAG_DIR + "flag")
        except:
            print("Sync Failed")
        
    def on_created(self, event):
        self.sync_directory()
        #self.upload_file(event)
        
    #def on_modified(self, event):
    #    self.upload_file(event)


class OnMyWatch:
    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        #self.observer.schedule(event_handler, WATCH_DIR, recursive = True)
        self.observer.schedule(event_handler, FLAG_DIR, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")
        self.observer.join()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s')
    watch = OnMyWatch()
    watch.run()


