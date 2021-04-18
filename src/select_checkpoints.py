from collections import defaultdict
import sys

import boto3


s3_client = boto3.client('s3')
S3_BUCKET = 'alchemists-uiuc-dlh-spring2021-us-east-2'
S3_PREFIX = 'flannel/checkpoint/'
CHECKPOINT_PATH = './explore_version_03/checkpoint/'

def get_response_iterator()
    paginator = client.get_paginator('list_objects_v2')
    return paginator.paginate(
        Bucket = S3_BUCKET,
        Prefix = S3_PREFIX)


def get_completed_models():
    completed_models = set()
    for response in get_response_iterator()
        for content in response['Contents']
            folder, filename = content['Key'].rsplit('/',1)
            if filename == 'log.png'
                completed_models.add(folder)
    return completed_models



def select_bestmodel():
    completed_models = get_completed_models()
    keys = set()
    for response in get_response_iterator():
        for content in response['Contents']:
            folder, filename = content['Key'].rsplit('/',1)
            if (folder in completed_models and 
                filename == 'model_best.pth.tar'):
                keys.append(content['Key'])
    return keys


def write_script():
    pass


def select_checkpoints():
    completed_models = get_completed_models()
    checkpoints = defaultdict(list)
    for response in response_iterator:
        for content in response['Contents']:
            folder, filename = content['Key'].rsplit('/',1)
            if folder not in completed_models:
                mtime = content['LastModified'].isoformat()
                checkpoints[folder].append((mtime, filename))

    final_checkpoint = {}
    for folder, fileset in checkpoints.items():
        filename = sorted(fileset,reverse=True)[0]
        final_checkpoint[folder] = filename



if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1]=='completed':
            select_bestmodel()
        elif sys.argv[1]=='status':
            pass
    else:
        select_checkpoints()
