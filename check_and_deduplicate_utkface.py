import os
import sys
import traceback
import datetime
import shutil
import signal
from collections import defaultdict

LOG_FILE = 'error_log.log'
STOP_FLAG_FILE = os.path.abspath("stop.flag")
if __name__ == "__main__" and os.environ.get("DEVELOPER_MODE") is None:
    DEVELOPER_MODE = True
else:
    DEVELOPER_MODE = os.environ.get("DEVELOPER_MODE", "0") == "1"

def dummy_handler(signum, frame):
    print("非命令行环境下，屏蔽Ctrl+C（KeyboardInterrupt），请通过UI停止去重。")

is_tty = sys.stdout.isatty()
if not is_tty:
    signal.signal(signal.SIGINT, dummy_handler)

def save_error_log(e):
    if DEVELOPER_MODE:
        err_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n[{err_time}] {repr(e)}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        print(f"发生错误，已记录到 {LOG_FILE}")
    sys.exit(1)

def is_image(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

def main():
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

    # 遍历所有文件夹，收集图片
    for folder in folders:
        if os.path.exists(STOP_FLAG_FILE):
            print("\n检测到停止标志，提前结束收集，直接进入整理流程。")
            os.remove(STOP_FLAG_FILE)
            break
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        save_error_log(e)