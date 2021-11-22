import hashlib
import os
import shutil

shutil.rmtree("img_filtered", True)
os.makedirs("img_filtered", exist_ok=True)
unique_files = dict()

files = os.listdir("img")
count = 0
for root, _, file_list in os.walk("img"):
    for f in file_list:
        if not f.endswith(".jpg"):
            continue
        fpath = os.path.join(root, f)
        count += 1
        data = open(fpath, "rb").read()
        md5 = hashlib.md5(data)
        if md5 in unique_files:
            print("replacing", unique_files[md5], "with", f)
        unique_files[md5] = f
        

print("Original", count)
print("Filtered", len(unique_files))