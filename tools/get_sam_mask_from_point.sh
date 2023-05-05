


# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/000100.npy  residence 000100
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/000400.npy  residence 000400
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/000800.npy  residence 000800
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/001500.npy  residence 001500
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/campus/campus-labels/train/sam_features/000100.npy  campus 000100
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/campus/campus-labels/train/sam_features/000400.npy  campus 000400
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/campus/campus-labels/train/sam_features/000800.npy  campus 000800
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/campus/campus-labels/train/sam_features/001500.npy  campus 001500
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/sci-art/sci-art-labels/train/sam_features/000100.npy  sci 000100
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/sci-art/sci-art-labels/train/sam_features/000400.npy  sci 000400
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/sci-art/sci-art-labels/train/sam_features/000800.npy  sci 000800
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/sci-art/sci-art-labels/train/sam_features/001500.npy  sci 001500
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/Mill19/building/building-labels/train/sam_features/000001.npy  building 000001
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/Mill19/building/building-labels/train/sam_features/000200.npy  building 000200
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/Mill19/building/building-labels/train/sam_features/000700.npy  building 000700
# python scripts/get_sam_mask_from_point.py   /data/yuqi/Datasets/MegaNeRF/Mill19/building/building-labels/train/sam_features/001200.npy  building 001200


ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/building_000001*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/building_000001.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/building_000200*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/building_000200.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/building_000700*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/building_000700.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/building_001200*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/building_001200.mp4  -y

ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/residence_000100*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/residence_000100.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/residence_000400*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/residence_000400.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/residence_000800*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/residence_000800.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/residence_001500*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/residence_001500.mp4  -y


ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/sci_000100*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/sci_000100.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/sci_000400*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/sci_000400.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/sci_000800*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/sci_000800.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/sci_001500*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/sci_001500.mp4  -y

ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/campus_000100*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/campus_000100.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/campus_000400*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/campus_000400.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/campus_000800*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/campus_000800.mp4  -y
ffmpeg -r 4 -pattern_type glob -i "/data/yuqi/code/GP-NeRF-semantic/zyq/campus_001500*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p zyq/campus_001500.mp4  -y


rm -rf /data/yuqi/code/GP-NeRF-semantic/zyq/*.png