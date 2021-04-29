import splitfolders
import header

splitfolders.ratio(
    "./data_preprocess/output/",
    output=header.data_dir,
    seed=1337,
    ratio=(0.7, 0.1, 0.2),
    group_prefix=2
)
