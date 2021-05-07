set -x
epochs=100
ck_n=50
workers=8
for i in $(seq 1 5); do
    python patched_flannel/classification/train.py --arch inception_v3 --epochs=$epochs --crop_size=299 -ck_n=$ck_n --cv=cv$i -j=$worker
done
aws s3 sync explore_version_03/checkpoint s3://alchemists-uiuc-dlh-spring2021-us-east-2/patched_flannel_1/checkpoint/
