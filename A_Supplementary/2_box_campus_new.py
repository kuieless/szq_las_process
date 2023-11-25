import cv2 
import os

# 输入视频路径列表
videos = [
    '/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/semantic_video/rgb_campus.mp4',
    '/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/semantic_video/M2F_campus.mp4', 
    '/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/semantic_video/semnerf_campus.mp4',
    '/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/semantic_video/ours_campus.mp4'
]

# 输出目录 
OUTPUT_DIR = '/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/semantic_video_output_new'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 帧率   
FPS = 24

# 暂停秒数
PAUSE_DURATION = 2
line_size=4
# 矩形坐标
X, Y, W, H = 270, 220, 300, 300

for video in videos:
    # 获取文件名
    name = os.path.basename(video)
    
    # 输出路径
    out_path = os.path.join(OUTPUT_DIR, name)

    # 读取输入、打开输出
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    size = (frame.shape[1], frame.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_path, fourcc, FPS, size)

    # 分段处理
    stop = 110
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        
        ret, frame = cap.read()
        
        if i < stop:
            out.write(frame)
            
        elif i == stop:
            for _ in range(10): 
                out.write(frame)
            # cv2.rectangle(frame, (X, Y), (X+W, Y+H), (0, 255, 255), line_size)
            # for _ in range(PAUSE_DURATION * FPS): 
            #     out.write(frame)
            # for _ in range(10): 
            #     out.write(frame)
            cv2.rectangle(frame, (X, Y), (X+W, Y+H), (0, 255, 255), line_size)
            out.write(frame)
        elif i>stop:
            pass

    cap.release()
    out.release()
    
print('处理完成!')