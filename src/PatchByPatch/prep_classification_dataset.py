import splitfolders

splitfolders.ratio(
    "./output",
    output="classification_data",
    seed=1337,
    ratio=(0.7, 0.1, 0.2),
    group_prefix=2
)
