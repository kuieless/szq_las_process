ffmpeg -i yingrenshi.mp4  -vf "crop=in_w:in_h-160" -c:a copy c_yingrenshi.mp4
ffmpeg -i campus.mp4  -vf "crop=in_w:in_h-160" -c:a copy c_campus.mp4
ffmpeg -i b1.mp4  -vf "crop=in_w:in_h-142" -c:a copy c_b1.mp4
ffmpeg -i b2.mp4  -vf "crop=in_w:in_h-142" -c:a copy c_b2.mp4
