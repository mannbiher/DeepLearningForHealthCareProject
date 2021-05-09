import os

import boto3
from tabulate import tabulate


client = boto3.client('s3')
S3_BUCKET = 'alchemists-uiuc-dlh-spring2021-us-east-2'
S3_CHECKPOINT = 'patched_results_v3/checkpoint/'
S3_RESULT = 'patched_results_v3/results/'


def get_response_iterator(prefix):
    paginator = client.get_paginator('list_objects_v2')
    return paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix=prefix)


def get_results():
    suffix = 'result_detail_'
    completed_models = set()
    for response in get_response_iterator(S3_RESULT):
        for content in response['Contents']:
            folder, filename = content['Key'].rsplit('/', 1)
            if (filename.startswith(suffix)
                and 'test' in filename
                    and filename.endswith('.csv')):
                model, _, fold = filename[len(
                    suffix):-len('.csv')].rsplit('_', 2)
                completed_models.add((model, fold))
    return completed_models


def get_checkpoints():
    suffix = 'f1_'
    trained_models = set()
    in_training_models = set()
    for response in get_response_iterator(S3_CHECKPOINT):
        for content in response['Contents']:
            folder, filename = content['Key'].rsplit('/', 1)
            if filename.startswith(suffix):
                model, _, fold = filename[len(
                    suffix):-len('.png')].rsplit('_', 2)
                trained_models.add((model, fold))
            if filename == 'model_best.pth.tar':
                model, fold = (folder.rsplit('/', 1)[-1]
                               .replace('20200407_patched_', '')
                               .rsplit('_', 1))

                in_training_models.add((model, fold))
    in_training_models = in_training_models - trained_models
    return trained_models, in_training_models


def print_table(table):
    table = sorted(table, key=lambda x: x[0]+x[1]+x[2])
    print(tabulate(table,
                   headers=['Model', 'Fold', 'Status'],
                   tablefmt='github'))


def get_progress():
    status_table = []
    completed_models = get_results()
    trained_models, in_training_models = get_checkpoints()
    in_progress_models = trained_models - completed_models
    for model, cv in in_progress_models:
        status_table.append([model, cv, 'Inference In-Progress'])
    for model, cv in in_training_models:
        status_table.append([model, cv, 'Training In-Progress'])
    for model, cv in completed_models:
        status_table.append([model, cv, 'Done'])

    print('Completed', len(completed_models))
    print('In-Progess', len(in_training_models | in_progress_models))
    return status_table


def write_script(commands, path):
    with open(path, 'w') as f:
        f.write('#!/usr/bin/env bash\n')
        for command in commands:
            f.write(command+'\n')


def create_download_script(path, folder_pattern="{}_20200407_patched_{}"):
    s3_cmd = 'aws s3 cp s3://{}/{} {}'
    suffix = 'result_detail_'
    commands = []
    for response in get_response_iterator(S3_RESULT):
        for content in response['Contents']:
            folder, filename = content['Key'].rsplit('/', 1)
            if (filename.startswith(suffix)
                    and filename.endswith('.csv')):
                model, _, fold = filename[len(
                    suffix):-len('.csv')].rsplit('_', 2)
                folder = folder_pattern.format(model, fold)
                copy_path = os.path.join(path, folder, filename)
                commands.append(s3_cmd.format(
                    S3_BUCKET, content['Key'], copy_path))
    write_script(commands, 'patched_download.sh')


if __name__ == '__main__':
    #table = get_progress()
    # print_table(table)
    create_download_script('patched_results/results/')
