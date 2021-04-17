import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        new_obj = event.src_path
        ### update these 2 lines
        checkpoint_dir = "/Users/sreddyasi/kgdata/"
        checkpoint_s3 = "s3://uiuc-dlh-spring2021-finalproject-us-east-2/checkpoint/"
        s3_key = new_obj.replace(checkpoint_dir,"")
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            # Event is created, you can process it now
            print("Watchdog received created event - % s." % new_obj)
            os.system("aws s3 cp {} {}{}".format(new_obj, checkpoint_s3, s3_key))
        elif event.event_type == 'modified':
            # Event is modified, you can process it now
            print("Watchdog received modified event - % s." % event.src_path)
            os.system("aws s3 cp {} {}{}".format(new_obj, checkpoint_s3, s3_key))


class OnMyWatch:
    # update this
    checkpoint_dir = "/Users/sreddyasi/kgdata/"
    def __init__(self):
        self.observer = Observer()
    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.checkpoint_dir, recursive = True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")
        self.observer.join()


watch = OnMyWatch()
watch.run()


