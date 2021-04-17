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
        
    def on_created(self, event):
        self.upload_file(event)
        
    def on_modified(self, event):
        self.upload_file(event)
  
class OnMyWatch:
    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, WATCH_DIR, recursive = True)
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


