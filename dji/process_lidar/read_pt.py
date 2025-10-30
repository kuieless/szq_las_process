# 脚本功能：
# 加载一个图像元数据文件(.pt)，并打印出其中 'meta_tags' 字典里所有的键(key)和值(value)。
# 这可以帮助您快速找到“曝光时间”等参数在您的数据中对应的确切键名。

import torch
from pathlib import Path
import sys
import os

# --- 项目路径设置 (与您的主脚本保持一致) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts


def main(hparams):
    dataset_path = Path(hparams.dataset_path)

    # --- 1. 查找一个示例元数据文件 ---
    # 我们只需要看一个文件，就能知道所有文件的结构
    try:
        # 优先从训练集中找第一个文件
        sample_metadata_path = sorted(list((dataset_path / 'train' / 'image_metadata').iterdir()))[0]
    except IndexError:
        print("错误: 在 'train/image_metadata' 文件夹中没有找到文件。")
        print("正在尝试从 'val/image_metadata' 文件夹中查找...")
        try:
            sample_metadata_path = sorted(list((dataset_path / 'val' / 'image_metadata').iterdir()))[0]
        except IndexError:
            print("错误: 在 'val/image_metadata' 文件夹中也没有找到文件。请检查您的数据集路径。")
            return

    print(f"正在读取示例元数据文件:\n  -> {sample_metadata_path}\n")

    # --- 2. 加载文件并打印所有键和值 ---
    try:
        metadata = torch.load(sample_metadata_path, weights_only=False)

        if 'meta_tags' in metadata:
            meta_tags = metadata['meta_tags']
            print("=" * 15 + " 'meta_tags' 中包含的键和值 " + "=" * 15)

            for key, value in meta_tags.items():
                # value可能不是简单的字符串，我们用.values来获取其内容
                try:
                    val_content = value.values
                except AttributeError:
                    val_content = value  # 如果没有.values属性，直接显示

                print(f"键 (Key)  : '{key}'")
                print(f"值 (Value): {val_content}\n" + "-" * 50)

            print("\n请从上面的列表中找到'曝光时间'对应的键名 (很可能是 'EXIF ExposureTime')，")
            print("然后将其填入您的主处理脚本 '1_process_las_by_exposure_time.py' 中。")

        else:
            print("错误: 在元数据文件中没有找到 'meta_tags' 字典。")

    except Exception as e:
        print(f"读取或处理文件时发生错误: {e}")


if __name__ == '__main__':
    # 这个脚本也使用和主脚本一样的参数传入方式，以获取 --dataset_path
    main(_get_opts())

