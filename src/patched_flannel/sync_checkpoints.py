"""Sync checkpoint to S3

This module contains code to sync checkpoints to S3. Unlike sync code
in cloud_utils folder, this has to be explicitly called from code.
This requires AWS CLI to be setup with AWS S3 credentials/IAM role.
"""
import os

S3_BUCKET = 'alchemists-uiuc-dlh-spring2021-us-east-2'
S3_PREFIX = 'flannel/checkpoint'

def s3_sync(checkpoint_dir, s3_prefix=S3_PREFIX):
    os.system(f"aws s3 sync {checkpoint_dir} s3://{S3_BUCKET}/{s3_prefix}")

