import glob

#phase = "train"
phase = "valid"

train_ratio = 0.5

topdir = "/data/MICCAI2025_RARE"
subdir1 = f"{topdir}/center_1/ndbe"
subdir2 = f"{topdir}/center_2/ndbe"

subdir3 = f"{topdir}/center_1/neo"
subdir4 = f"{topdir}/center_2/neo"

for subdir in [subdir1, subdir2]:
    files = sorted(glob.glob(f"{subdir}/*.png"))
    num_files = int(len(files) * train_ratio)

    if phase == "train":
        for file in files[:num_files]:
            print(f"{file.replace(topdir+'/', "")},0")
    elif phase == "valid":
        for file in files[num_files:]:
            print(f"{file.replace(topdir+'/', "")},0")

if phase == "valid":
    for subdir in [subdir3, subdir4]:
        files = sorted(glob.glob(f"{subdir}/*.png"))
    
        for file in files:
            print(f"{file.replace(topdir+'/', "")},1")
