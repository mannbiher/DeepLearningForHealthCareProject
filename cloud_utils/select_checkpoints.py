from collections import defaultdict
import sys
import os

import boto3
from tabulate import tabulate

client = boto3.client('s3')
S3_BUCKET = 'alchemists-uiuc-dlh-spring2021-us-east-2'
S3_PREFIX = 'flannel/checkpoint/'
CHECKPOINT_PATH = '../src/explore_version_03/checkpoint/'


def get_response_iterator():
    paginator = client.get_paginator('list_objects_v2')
    return paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX)


def get_completed_models():
    completed_models = set()
    for response in get_response_iterator():
        for content in response['Contents']:
            folder, filename = content['Key'].rsplit('/', 1)
            if filename == 'log.png':
                completed_models.add(folder)
    return completed_models


def select_bestmodel():
    completed_models = get_completed_models()
    best_models = {}
    for response in get_response_iterator():
        for content in response['Contents']:
            folder, filename = content['Key'].rsplit('/', 1)
            if (folder in completed_models and
                    filename == 'model_best.pth.tar'):
                best_models[folder] = filename
    return best_models


def parse_parameters(basedir):
    folder_parts = basedir.rsplit('/', 1)[-1].split('_')
    model = folder_parts[0]
    cv = folder_parts[-1]
    return model, cv


def get_status_table(folders, is_completed=False):
    status = 'In-Progress'
    if is_completed:
        status = 'Done'
    table = []
    for folder in folders:
        model, cv = parse_parameters(folder)
        table.append([model, cv, status])
    return table


def print_table(table):
    table = sorted(table, key=lambda x: x[0]+x[1]+x[2])
    print(tabulate(table,
                   headers=['Model', 'Fold', 'Status'],
                   tablefmt='github'))


def print_status():
    completed = list(get_completed_models())
    in_progress = list(select_checkpoints().keys())
    table = get_status_table(completed, is_completed=True)
    table += get_status_table(in_progress)
    print_table(table)


def write_script(commands, path):
    with open(path, 'w') as f:
        f.write('#!/usr/bin/env bash\n')
        for command in commands:
            f.write(command+'\n')


def download_completed(model_filter=None):
    s3_cmd = 'aws s3 cp s3://{}/{} {}'
    checkpoints = select_bestmodel()
    if model_filter:
        checkpoints = {k: v for k, v in checkpoints.items()
            if model_filter in k}
    print_table(get_status_table(checkpoints.keys(),is_completed=True))
    commands = []
    for folder, filename in checkpoints.items():
        localpath = CHECKPOINT_PATH + \
            folder.replace(S3_PREFIX, '') + '/' + filename
        s3_key = folder+'/'+filename
        commands += [s3_cmd.format(S3_BUCKET, s3_key, localpath)]
    write_script(commands, 'bestmodel_download.sh')




def download_inprogress(model_filter=None):
    s3_cmd = 'aws s3 cp s3://{}/{} {}'
    checkpoints = select_checkpoints()
    if model_filter:
        checkpoints = {k: v for k, v in checkpoints.items()
            if model_filter in k}
    print_table(get_status_table(checkpoints.keys()))

    commands = []
    for folder, filename in checkpoints.items():
        localpath = CHECKPOINT_PATH + \
            folder.replace(S3_PREFIX, '') + '/' + filename
        s3_key = folder+'/'+filename
        s3_key_log = folder + '/log.txt'
        localpath_log = CHECKPOINT_PATH + \
            folder.replace(S3_PREFIX, '') + '/log.txt'
        commands += [s3_cmd.format(S3_BUCKET, s3_key, localpath),
                     s3_cmd.format(S3_BUCKET, s3_key_log, localpath_log)]
    write_script(commands, 'resume_download.sh')


def select_checkpoints():
    completed_models = get_completed_models()
    checkpoints = defaultdict(list)
    for response in get_response_iterator():
        for content in response['Contents']:
            folder, filename = content['Key'].rsplit('/', 1)
            if (folder not in completed_models
                    and filename.endswith('.pth.tar')):
                mtime = content['LastModified'].isoformat()
                checkpoints[folder].append((mtime, filename))

    final_checkpoint = {}
    for folder, fileset in checkpoints.items():
        filename = sorted(fileset, reverse=True)[0]
        final_checkpoint[folder] = filename[1]
    # print(final_checkpoint)
    return final_checkpoint


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'status':
            print_status()
        elif sys.argv[1] == 'completed':
            download_completed()
        elif sys.argv[1] == 'inprogress':
            download_inprogress()
    elif len(sys.argv) == 3:
        if sys.argv[1]=='inprogress':
            download_inprogress(sys.argv[2])
        elif sys.argv[1] == 'completed':
            download_completed(sys.argv[2])
    else:
        print('usage: python select_checkpoints.py (completed|status|inprogress) [model]')
        
