
#!/bin/bash

original_images_path="/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/original_image"

subfolders=($(ls -d "${original_images_path}"/*/ | sort))

counts_list=()
counts_list+=(0)
# 遍历一级子文件夹，获取图片数量并输出
for subfolder in "${subfolders[@]}"; do
    count=$(ls -l "$subfolder"/*.JPG | wc -l)
    echo "Folder $subfolder has $count images."
    ((total_count += count))
    counts_list+=($total_count)
done

echo "${counts_list[@]}"
# echo "${counts_list[1+1]}"


original_images_list_json_path='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/original_image_list.json'
infos_path='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/BlocksExchangeUndistortAT.xml'
dataset_path='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan'
debug=False
start=counts_list[1]

# custom_numbers=(1 2 3 5 6)
# for ((i = 1; i <= ${#custom_numbers[@]}; i++)); do
custom_numbers=(1 2)
for ((i = 2; i <= ${#custom_numbers[@]}; i++)); do
    index=$((i - 1))

    start=${counts_list[$((i-1))]}
    end=${counts_list[$i]}
    las_path=/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/H${custom_numbers[$index]}.las
    las_output_path=/data/yuqi/code/GP-NeRF-semantic/dji/process_las/output_${custom_numbers[$index]}
    output_path=/data/yuqi/code/GP-NeRF-semantic/dji/process_las/output_${custom_numbers[$index]}

    echo $start
    echo $end
    echo $las_path
    
    # python dji/process_las/1_process_each_line_las.py     \
    #     --dataset_path  $dataset_path  --las_path  $las_path       --las_output_path   $las_output_path   \
    #     --start  $start   --end   $end

    python dji/process_las/2_convert_lidar_2_depth_color.py     \
        --original_images_path  $original_images_path  --original_images_list_json_path  $original_images_list_json_path  \
        --infos_path  $infos_path  --dataset_path  $dataset_path  --output_path  $output_path     \
        --debug  $debug     --start  $start   --end   $end     --las_output_path  $las_output_path

done
