import exifread


def get_image_timestamps(image_path):
    """
    读取并打印图像的详细时间戳信息
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)

            if not tags:
                print(f"在 {image_path} 中没有找到 EXIF 数据。")
                return

            print(f"--- 图像: {image_path} 的时间戳信息 ---")

            # 查找关键的时间标签
            datetime_original = tags.get('EXIF DateTimeOriginal')
            subsec_original = tags.get('EXIF SubSecTimeOriginal')

            if datetime_original:
                # 组合成完整的精细时间戳
                fine_grained_time = str(datetime_original.values)
                if subsec_original:
                    fine_grained_time += f".{subsec_original.values}"

                print(f"  - 原始拍摄时间 (DateTimeOriginal): {datetime_original}")
                if subsec_original:
                    print(f"  - 亚秒级精度 (SubSecTimeOriginal): {subsec_original}")
                    print(f"  => 精确拍摄时间: {fine_grained_time}")
                else:
                    print("  - 未找到亚秒级精度时间戳。")

            else:
                print("  - 未找到原始拍摄时间 (DateTimeOriginal)。")

            # 打印其他可能存在的时间戳
            for tag, value in tags.items():
                if 'date' in tag.lower() or 'time' in tag.lower():
                    # 避免重复打印已经处理过的标签
                    if tag not in ['EXIF DateTimeOriginal', 'EXIF SubSecTimeOriginal']:
                        print(f"  - 其他时间相关标签: {tag} = {value}")

    except Exception as e:
        print(f"读取文件时出错: {e}")


# --- 使用方法 ---
# 将 'path/to/your/image.jpg' 替换为你的图片路径
# 在 Windows 上路径可能是 'C:\\Users\\YourUser\\Pictures\\photo.jpg'
get_image_timestamps('F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\DJI_20231116105324_0002_Zenmuse-L1-mission.JPG')