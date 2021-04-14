# TODO

1. Raman/Srikanth review patch-by patch paper. Find if any source code is available.
2. Raman can work on Segment Model/code.
3. Srikanth would research on models that would be useful for patch-by-patch. (Follow up in OH as well)
4. Me and Satish can setting up AWS and access keys for everyone.
5. Running the FLANNEL code as it is
6. We have to agree on AWS pricing.
We will run the GPU instance.
7. Setup a s3 bucket, to store the checkpoints, models, input files, output files and so on

# Plan (Didn't work)

~~- 04/03 (Decide on base models, segment model, RUn base flannel)~~
~~- 04/06 (Run FLannel improvemnet, complete base flannel evaluation)~~
~~- 04/10 (Complete Flannel improvment evaluation, decide on what to go in draft)~~
~~- 04/13 (First version of draft)~~

# Plan

1. Base flannel graph so that we have some data for the draft report
2. Draft sections split work among us
3. Segmentation has to be done on in-scope images
4. Model details for the segmentation
5. two models in parallel. inception_v3 (Device 2 received some error. Log_max not defined.)
6. autoscaling group => 1 spot instance => every time => 1 spot
   Code => get the epoch that was running and pass resume option in the code (Argment)
   -----
   Checkpoints ===> models into s3
   ---
   --resume --epoch 
   ------
   Tomorrow we should have a decision to use spot or not




