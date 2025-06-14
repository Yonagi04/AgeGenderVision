import os
import shutil
from collections import defaultdict

# 需要处理的文件夹
folders = [
    r'data/UTKFace/archive/UTKFace',
    r'data/UTKFace/archive/crop_part1',
    r'data/UTKFace/archive/utkface_aligned_cropped/UTKFace',
    r'data/UTKFace/archive/utkface_aligned_cropped/crop_part1',
]

output_dir = r'data/UTKFace/cleaned'
os.makedirs(output_dir, exist_ok=True)

all_files = []
name_to_path = dict()
duplicate_count = 0

def is_image(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

# 遍历所有文件夹，收集图片
for folder in folders:
    if not os.path.exists(folder):
        continue
    for fname in os.listdir(folder):
        if is_image(fname):
            fpath = os.path.join(folder, fname)
            all_files.append(fpath)
            # 以文件名为唯一标识，若已存在则计为重复
            if fname not in name_to_path:
                name_to_path[fname] = fpath
            else:
                duplicate_count += 1

print(f'原始图片总数: {len(all_files)}')
print(f'去重后图片数: {len(name_to_path)}')
print(f'重复图片数: {duplicate_count}')

# 复制唯一图片到cleaned目录
for fname, src_path in name_to_path.items():
    dst_path = os.path.join(output_dir, fname)
    shutil.copy2(src_path, dst_path)

print(f'所有唯一图片已复制到: {output_dir}') 