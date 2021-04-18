from collections import defaultdict
import torch


CHECKPOINT_PATH = './explore_version_03/checkpoint/'

s3_client = 

def list_all_checkpoints(folder):
    checkpoint_map = defaultdict(list)
    for basedir, dirs, files in os.walk(folder):
        # do not check files which are already completed
        if 'log.png' in files:
            continue
        for filename in files:
            if filename.endswith('.pth.tar'):
                checkpoint_map[basedir].append(filename)

    return checkpoint_map
    

def get_max_checkpoint(checkpoints):
    max_checkpoint = {}
    for folder, files in checkpoints.items():
        max_checkpoint[folder]= max(int(filename.split('.',1)[0])
        for filename in files
        if filename != 'model_best.pth.tar')
    return max_checkpoint
    

def parse_parameters(basedir, epoch):
    folder_parts = basedir.rsplit('/',1)[-1].split('_')
    model = folder_parts[0]
    cv = folder_parts[-1]
    return f'python FLANNEL/ensemble_step1.py --arch {model} --epochs=200 -ck_n=50 --cv={cv} -j=6 -r={epoch}'


def restore():
    checkpoints = list_all_checkpoints(CHECKPOINT_PATH)
    print(checkpoints)
    max_checkpoint = get_max_checkpoint(checkpoints)
    print(max_checkpoint)
    for folder, files in checkpoints.items():
        for filename in files:
            if filename != 'model_best.pth.tar':
                continue
            checkpoint = torch.load(file_)
            epoch = int(checkpoint['epoch'])
            if epoch > max_checkpoint.get(folder,-1):
                command = parse_parameters(folder, epoch)
                print('resume command', command)
                shutil.copyfile(filepath,
                    os.path.join(folder, f'{epoch}.checkpoint.pth.tar'))

if __name__ == '__main__':
    restore()