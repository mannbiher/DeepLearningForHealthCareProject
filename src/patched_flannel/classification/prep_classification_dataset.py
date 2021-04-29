import splitfolders
import header
from src.data_preprocess.segmentation import header as seg_header

splitfolders.ratio(
    seg_header.dir_save,
    output=header.data_dir,
    seed=1337,
    ratio=(0.7, 0.1, 0.2),
    group_prefix=2
)
