# general libs
import cv2
import ffmpeg
import numpy as np
import time
import os,subprocess,shutil
import sys
import argparse
from PIL import ImageDraw,ImageFont,Image
import copy
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))


def get_arguments():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("-i1","--input1",type=str,required=True,help='第一视频路径')
    parser.add_argument("-i2","--input2",type=str,required=True,help='第二视频路径')
    parser.add_argument("-bb","--bbox",type=str,required=False,help='归一化的裁剪区域: x,y-w,h')
    parser.add_argument("-t","--time",type=str,required=False,help='截帧时间：hh:mm:ss.ms')
    parser.add_argument("-ts","--timespan",type=str,required=False,help='裁剪时间段：hh:mm:ss.ms-hh:mm:ss.ms')
    parser.add_argument("-o","--output",type=str,required=True,help='输出文件夹')
    parser.add_argument("-ss","--samesize",required=False,action='store_true',\
        help='如果设置-ss，则较小的视频输出将被调整到和较大的视频输出同样的大小')
    parser.add_argument("-ps","--print_speed",required=False,action='store_true',\
        help='如果设置-ss，则较小的视频输出将被调整到和较大的视频输出同样的大小')
    parser.add_argument("-sf","--samefps",required=False,action='store_true',\
        help='如果设置-sf，则第一视频的帧率会向第二视频看齐')
    parser.add_argument("-b1","--bitrate1",type=float,required=False,\
        help='有时需要将视频1降质再和视频2对比，因此需要设置视频1输出的归一化码率：-b1：0.5')
    parser.add_argument("-fb","--framebias",type=float,required=False,\
        help='与-t或-ts配合使用，有时ffmpeg的裁剪时间会有1-2帧的偏差，为了使两个视频的输出时间完全对齐，\
        需要调整第一视频的时间参数，比如：\"-fb 0.04\" 会将第一视频的裁剪时间延后0.04秒。\
        请注意，如果不设置裁减时间或者裁剪时间从0开始，则负值-fb不起作用')
    parser.add_argument("-c","--compare",type=str,required=False,\
        help='如果设置-c，则输出一个对比动画。由于对比视频要求两个输出的尺寸一致，-c会强制设置-ss。\
        -c的参数定义了\"镜头级\"剧本大纲，镜头间用\'-\'连接。可选镜头包括：\
        播放视频1：\'s1\'，播放视频2：\'s2\'，\
        左右完整视频对比：\'s1s2\'，上下完整视频对比：\'s12s\'，\
        左右分割对比：\'s1vs2\'，上下分割对比：\'s1v2s\'，\
        左右视频扫描对比：\'s1vws2\'，上下视频扫描对比：\'s1vw2s\'，左右视频反向扫描对比：\'s1vws2r\'，上下视频反向扫描对比：\'s1vw2sr\'，\
        左右图像扫描对比：\'s1iws2\'，上下图像扫描对比：\'s1iw2s\'，左右图像反向扫描对比：\'s1iws2r\'，上下图像反向扫描对比：\'s1iw2sr\'，。\
        每个镜头名后可以加参数，参数与镜头名以及参数与参数之间用\',\'隔开。包括：\
        镜头总时长（单位秒）：\'t{}\'，分割线位置：\'L{}\'；扫描线停留位置和停留时间：\'l{}:{}\' 或\'l{}:{},b\'或\'l{}:{},s\'，\
        \',b\'表示扫描线会从停留位置返回到起始位置，\',s\'表示扫描从扫描线停留位置开始；\
        帧率减速倍率：\'r{}\'，如r0.25表示每帧会重复编码四次，即1/4播放速度，比如一个左右视频扫描对比镜头，希望扫描线停留在0.6的位置3秒钟：s1vws2,l0.6:3\
        ')
    parser.add_argument("-l","--label",type=str,required=False,\
        help='与-c配合使用，设置视频1和视频2的英文角标文字：-l 原始画面-AI增强')
    parser.add_argument("-lp","--labelposition",type=str,required=False,\
        help='与-l配合使用，改变默认的label位置，通常不需要人工设置。可选位置为九宫格：由n个由连词符‘-’连接的双字子字符串组成：\
        双字子串中第一个字符是上下控制符：t（top），m（中），b（下）；第二个字符是左右控制符：l（左），m（中），r（右）\
        比如：\'bl-br\'，表示文本内容text会出现在左下和右下')
    parser.add_argument("-bo","--bitrateOutput",type=float,required=False,\
        help='与-c配合使用，设置输出视频的码率，比如5mb/s：\'-bo 5000000\'')
    parser.add_argument("-fr","--frameRate",type=float,default=-1) # overwrite framerate for individual outputs
    parser.add_argument("-crf","--crf",type=float,default=18)
    parser.add_argument("-suf","--suffix",type=str,default='')
    return parser.parse_args()
    
def get_bbox(bboxStr):
    x=0.0
    y=0.0
    w=1.0
    h=1.0
    tmp=bboxStr.split("-")
    if len(tmp) == 2:
        x,y = tmp[0].split(",")
        w,h = tmp[1].split(",")
    else:
        print("bbox format incorrect in {}!".format(bboxStr))
        sys.exit(1)
    return np.array([x,y,w,h]).astype(np.float)

def get_timespan(timeSpanStr):
    return separate_str(str(timeSpanStr))

def separate_str(str):
    tmp=str.split("-")
    if len(tmp) == 2:
        return tmp
    else:
        print("separate_str failed in {}!".format(timeSpanStr))
        sys.exit(1)

def alter_time(timeStr,bias):
    tmp=np.array(timeStr.split(":")).astype(float)
    tmp[2] = tmp[2] + bias
    if tmp[2] < 0:
        tmp[1] = tmp[1] - 1
        tmp[2] = tmp[2] + 60
        if tmp[1] < 0:
            tmp[0] = tmp[0] - 1
            tmp[1] = tmp[1] + 60
    if tmp[2] >= 60:
        tmp[1] = tmp[1] + 1
        tmp[2] = tmp[2] - 60
        if tmp[1] >= 60:
            tmp[0] = tmp[0] + 1
            tmp[1] = tmp[1] - 60
    if tmp[0] < 0 or tmp[1] < 0 or tmp[2] < 0:
        print('warning: because of framebias, some of your time calculation results in negative time. Please try a -ts beginning from positive time.')
    return "{}:{}:{}".format(int(tmp[0]),int(tmp[1]),tmp[2])

def get_resolution(videoFn):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', videoFn]
    status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if status.returncode == 0:
        result = status.stdout.decode("utf-8").split('\n')[0].split("x")
        if len(result) == 2:
            return np.array(result).astype(np.float)
        else:
            print("get_resolution failed to parse output: {}".format(result))
            return (0, 0)
    else:
        print("get_resolution failed {}".format(status.stdout.decode("utf-8")))
        return (0, 0)

def get_framenumber(videoFn):
    if videoFn.lower().endswith(('.png', '.jpg', '.jpeg')):
        return len(glob.glob(os.path.join(os.path.split(videoFn)[0], '*' + os.path.splitext(os.path.split(videoFn)[1])[1])))
    else:
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'default=noprint_wrappers=1:nokey=1', videoFn]
        status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if status.returncode == 0:
            return int(status.stdout.decode("utf-8"))
        else:
            print("get_framenumber failed {}".format(status.stdout.decode("utf-8")))
            return 0

def get_videobitrate(videoFn):
    # if input is image(s), return default bitrate
    if videoFn.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 15000000
    else:
        command = ['ffprobe', '-v', 'error', '-show_entries', 'format=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1', videoFn]
        status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if status.returncode == 0:
            return float(status.stdout.decode("utf-8"))
        else:
            print("get_videobitrate failed {}".format(status.stdout.decode("utf-8")))
            return 0

def get_videoduration(videoFn):
    # if input is image(s), return default video duration
    if videoFn.lower().endswith(('.png', '.jpg', '.jpeg')):
        return get_framenumber(videoFn) / get_framerate(videoFn)
    else:
        command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', videoFn]
        status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if status.returncode == 0:
            return float(status.stdout.decode("utf-8"))
        else:
            print("get_videoduration failed {}".format(status.stdout.decode("utf-8")))
            return 0

def get_framerate(videoFn):
    # if input is image(s), return default framerate
    if videoFn.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 25
    else:
        command = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', videoFn]
        status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if status.returncode == 0:
            raw = status.stdout.decode("utf-8").split('/')
            return float(raw[0])/float(raw[1])
        else:
            print("get_framerate failed {}".format(status.stdout.decode("utf-8")))
            return 0

def get_timeinsecond(timeStr):
    tmp=np.array(timeStr.split(":")).astype(float)
    t = tmp[0]
    for i in range(1,len(tmp)):
        t = tmp[i] + t * 60
    return t

def cut_video(input,output,t,t2='',resizeRatio=1,bitrateRatio=1,frameRateToSync=0,padding=[0,0],crf=18):
    size = get_resolution(input)
    command = ['ffmpeg', '-i', input, '-ss', t, '-crf', str(crf)]
    if bitrateRatio<1:
        command.extend(['-b:v', str(get_videobitrate(input)*bitrateRatio)])
    if frameRateToSync>0:
        command.extend(['-r', str(frameRateToSync)])
    if t2:
        command.extend(['-to', t2])
    else:
        command.extend(['-vframes', '1'])
    if resizeRatio!=1 or padding!=[0,0]:
        filterStr = ''
        if resizeRatio!=1:
            filterStr += 'scale={}:{}'.format(int(size[0] * resizeRatio),int(size[1] * resizeRatio))
        if padding!=[0,0]:
            if len(filterStr)>1:
                filterStr+=','
            filterStr += 'pad={}:{}:{}:{}'.format(int(size[0]*resizeRatio+padding[0]),int(size[1]*resizeRatio+padding[1]),int(padding[0]/2),int(padding[1]/2))
        command.extend(['-filter:v',filterStr])
    command.extend(['-y', output])
    status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if status.returncode == 0:
        return True,output
    else:
        print("cut_video failed {}".format(status.stdout.decode("utf-8")))
        return False

def resize_video(input,output,resizeRatio=1,bitrateRatio=1,frameRateToSync=0,padding=[0,0], crf=18):
    if resizeRatio != 1 or bitrateRatio != 1 or padding != [0,0]:
        size = get_resolution(input)
        command = ['ffmpeg', '-i', input, 'crf', str(crf), '-y']
        if bitrateRatio<1:
            command.extend(['-b:v', str(get_videobitrate(input)*bitrateRatio)])
        if resizeRatio!=1 or padding!=[0,0]:
            filterStr = ''
            if resizeRatio!=1:
                filterStr += 'scale={}:{}'.format(int(size[0] * resizeRatio),int(size[1] * resizeRatio))
            if padding!=[0,0]:
                if len(filterStr)>1:
                    filterStr+=','
                filterStr += 'pad={}:{}:{}:{}'.format(int(size[0]*resizeRatio+padding[0]),int(size[1]*resizeRatio+padding[1]),int(padding[0]/2),int(padding[1]/2))
            command.extend(['-filter:v',filterStr])
        if frameRateToSync>0:
            command.extend(['-r', str(frameRateToSync)])
        command.append(output)
        status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if status.returncode == 0:
            return True,output
        else:
            print("resize_video failed {}".format(status.stdout.decode("utf-8")))
            return False
    else:
        shutil.copyfile(input,output)
        return True,output

def crop_video(input,output,bbox,resizeRatio=1,bitrateRatio=1,frameRateToSync=0, crf=18):
    size = get_resolution(input)
    bboxP = np.array([bbox[0] * size[0], bbox[1] * size[1], bbox[2] * size[0], bbox[3] * size[1]]).astype(np.int)
    if resizeRatio != 1:
        resizeStr = ',scale={}:{}'.format(int(bboxP[2]*resizeRatio),int(bboxP[3]*resizeRatio))
    else:
        resizeStr = ''
    command = ['ffmpeg', '-i', input, '-crf', str(crf), '-filter:v', 
                   "crop={}:{}:{}:{}{}".format(bboxP[2],bboxP[3],bboxP[0],bboxP[1],resizeStr), '-y']
    if bitrateRatio<1:
        command.extend(['-b:v', str(get_videobitrate(input)*bitrateRatio)])
    if frameRateToSync>0:
        command.extend(['-r', str(frameRateToSync)])
    command.append(output)
    status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if status.returncode == 0:
        return True,output
    else:
        print("crop_video failed {}".format(status.stdout.decode("utf-8")))
        return False

def cut_crop_video(input,output,bbox,t,t2='',resizeRatio=1,bitrateRatio=1,frameRateToSync=0,crf=18):
    size = get_resolution(input)
    bboxP = np.array([bbox[0] * size[0], bbox[1] * size[1], bbox[2] * size[0], bbox[3] * size[1]]).astype(np.int)
    if resizeRatio != 1:
        resizeStr = ',scale={}:{}'.format(int(bboxP[2]*resizeRatio),int(bboxP[3]*resizeRatio))
    else:
        resizeStr = ''
    if t2:
        command = ['ffmpeg', '-i', input, '-crf', str(crf), '-ss', t, '-to', t2, '-filter:v', 
                   "crop={}:{}:{}:{}{}".format(bboxP[2],bboxP[3],bboxP[0],bboxP[1],resizeStr), '-y']
    else:
        command = ['ffmpeg', '-i', input, '-crf', str(crf), '-ss', t, '-vframes', '1', '-filter:v', 
                   "crop={}:{}:{}:{}{}".format(bboxP[2],bboxP[3],bboxP[0],bboxP[1],resizeStr), '-y']
    if bitrateRatio<1:
        command.extend(['-b:v', str(get_videobitrate(input)*bitrateRatio)])
    if frameRateToSync>0:
        command.extend(['-r', str(frameRateToSync)])
    command.append(output)
    status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if status.returncode == 0:
        return True,output
    else:
        print("cut_crop_video failed: {}".format(status.stdout.decode("utf-8")))
        return False

#position 是一个字符串。由n个由连词符‘-’连接的双字子字符串组成：
#双字子串中第一个字符是上下控制符：t（top），m（中），b（下）；第二个字符是左右控制符：l（左），m（中），r（右）
#比如：position='bl-br'，表示文本内容text会出现在左下和右下
def draw_text(frame,text,position='bl-br',color=(255,255,255)):
    fontSizeR = 0.033
    w = frame.shape[1]
    h = frame.shape[0]
    ############################################
    # work around font path issue, assuming the required font exists in the current file's directory
    cwd=os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #fontFile = "arial.ttf" #
    fontFile = 'Alibaba-PuHuiTi-Regular.otf'
    font = ImageFont.truetype(fontFile, int(min(w,h)*fontSizeR))
    cwd=os.chdir(cwd)
    ############################################
    image = Image.fromarray(frame, 'RGB')
    draw = ImageDraw.Draw(image)
    draw.font = font
    ts = draw.textsize(text)
    ts = (ts[0],max(ts[1],draw.textsize('gh')[1]))
    b = ts[1] * 0.2
    shadowSize = max(1,font.size/20)
    poses = position.split('-')
    for pos in poses:
        if pos[0] == 't':
            y = h*0.01
        elif pos[0] == 'b':
            y = h*(1-0.01)-ts[1]
        else:# m
            y = (h-ts[1])/2
        if pos[1] == 'l':
            x = w*0.01
        elif pos[1] == 'r':
            x = w*(1-0.01)-ts[0]
        else:# m
            x = (w-ts[0])/2
        draw.rectangle([(x-b,y-b),(x+ts[0]+b,y+ts[1]+b)],fill=(36,90,241))
        draw.text((x,y),text,fill=(0,0,0),font=font)
        draw.text((x+shadowSize,y),text,fill=color,font=font)
    frame = copy.deepcopy(np.ascontiguousarray(np.asarray(image).reshape(image.size[1], image.size[0], 3)))
    return frame

class Reader:
    frame1=None
    frame2=None
    w=0
    h=0
    fps=0
    bitrate=0
    def __init__(self,input1,input2): #constructor
        self.reader1=None
        self.reader2=None
        if input1.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.frame1 = cv2.imread(input1)
            self.frame2 = cv2.imread(input2)
            self.w = self.frame1.shape[1]
            self.h = self.frame1.shape[0]
            w2 = self.frame2.shape[1]
            h2 = self.frame2.shape[0]
            if self.w != w2 or self.h != h2:
                print('Frame sizes are not identical. Try -ss option to upsample the smaller one if they are in the same aspect ratio.')
                exit(1)
            self.fps = 25 #default fps
            self.bitrate = 8000000
        else:
            self.reader1 = cv2.VideoCapture(input1)
            self.reader2 = cv2.VideoCapture(input2)
            if not self.reader1.isOpened():
                print("fail to open video in {}".format(input1))
                sys.exit(1)
            if not self.reader2.isOpened():
                print("fail to open video in {}".format(input2))
                sys.exit(1)
            self.w = int(self.reader1.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
            self.h = int(self.reader1.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
            self.fps = self.reader1.get(cv2.CAP_PROP_FPS)
            w2 = int(self.reader2.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
            h2 = int(self.reader2.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
            fps2 = self.reader2.get(cv2.CAP_PROP_FPS)
            if self.w != w2 or self.h != h2:
                print('Frame sizes are not identical. Try -ss option to upsample the smaller one if they are in the same aspect ratio.')
                exit(1)
            self.bitrate = get_videobitrate(input1)
            bitrate2 = get_videobitrate(input2)
            self.bitrate = self.bitrate if self.bitrate > bitrate2 else bitrate2
    def getNextFramePairs(self):
        if self.frame1 is not None:
            return self.frame1,self.frame2
        elif self.reader1 is not None and self.reader1.isOpened():
            succeed1,frame1 = self.reader1.read()
            succeed2,frame2 = self.reader2.read()
            if not succeed1 or not succeed2:
                return None,None
            else:
                return frame1,frame2
        else:
            print('Reader has not been opened.')
            exit(1)
    def __del__(self):
        if self.reader1 is not None and self.reader1.isOpened():
            self.reader1.release()
        if self.reader1 is not None and self.reader2.isOpened():
            self.reader2.release()

class ScriptInterpreter:
    lrFull = False
    udFull = False
    lrSwipe = False
    udSwipe = False
    labelPos = None
    def __init__(self,shots,labelPos=None):
        if labelPos is not None:
            self.labelPos = labelPos
        for shot in shots:
            cmds = shot.split(',')
            if cmds[0] == 's1s2':
                self.lrFull = True
            elif cmds[0] == 's12s':
                self.udFull = True
            elif cmds[0].find('s2') > 0:
                self.lrSwipe = True
            elif cmds[0].find('2s') > 0:
                self.udSwipe = True

import atexit
class Writer:
    frame = None# The render buffer. Do not reallocate memory!
    def __init__(self,r,si,output,bitrate=None,crf=18): #constructor
        if si.lrFull and si.udFull:
            print('s1s2 and s12s cannot coexist, please modify your shot script.')
            exit(1)
        elif si.lrFull:
            w = r.w * 2
            h = r.h
        elif si.udFull:
            w = r.w
            h = r.h * 2
        else:
            w = r.w
            h = r.h
        w = w if w%2==0 else w+1
        h = h if h%2==0 else h+1
        br = bitrate if bitrate else r.bitrate
        # self.writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(w, h), r='%.02f' % r.fps)
        #                     .output(output, vcodec='libx264', pix_fmt='yuv420p', crf=str(crf), video_bitrate=str(int(br/1000))+'k', r='%.02f' % r.fps)
        #                     .overwrite_output()
        #                     .run_async(pipe_stdin=True)
        #                 )
        self.writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(w, h), r='%.02f' % r.fps)
                            .output(output, pix_fmt='yuv420p', video_bitrate=str(int(br/1000))+'k', r='%.02f' % r.fps)
                            .overwrite_output()
                            .run_async(pipe_stdin=True)
                        )

        # self.writer = (ffmpeg
        #                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(w, h), r='%.02f' % r.fps)
        #                .output(output, vcodec='libx264', pix_fmt='yuv420p', crf=str(crf), video_bitrate=str(int(br/1000))+'k', r='%.02f' % r.fps)
        #                .overwrite_output()
        #                .run_async(pipe_stdin=True)
        #                )
        self.frame = np.zeros((h,w,3),np.uint8)
    
    def write(self):
        if self.writer.stdin.closed:
            print("Warning: Attempted to write to a closed stdin pipe.")
            return

        self.writer.stdin.write(self.frame.astype(np.uint8).tobytes())

    def close_writer(self):
        if hasattr(self, 'writer') and self.writer:
            self.writer.stdin.close()
            self.writer.wait()

    def __del__(self):
        # Avoid relying on __del__ for cleanup, use the atexit-registered method
        pass


def addLabel(si,labels,frame1=None,frame2=None):
    if not labels or len(labels) < 2:
        return frame1, frame2
    if si.lrSwipe:
        if frame1 is not None:
            frame1 = draw_text(frame1,labels[0],si.labelPos if si.labelPos else 'bl-br')
        if frame2 is not None:
            frame2 = draw_text(frame2,labels[1],si.labelPos if si.labelPos else 'bl-br')
    elif si.udSwipe:
        if frame1 is not None:
            frame1 = draw_text(frame1,labels[0],si.labelPos if si.labelPos else 'tl-bl')
        if frame2 is not None:
            frame2 = draw_text(frame2,labels[1],si.labelPos if si.labelPos else 'tl-bl')
    elif si.lrFull:
        if frame1 is not None:
            frame1 = draw_text(frame1,labels[0],si.labelPos if si.labelPos else 'bl')
        if frame2 is not None:
            frame2 = draw_text(frame2,labels[1],si.labelPos if si.labelPos else 'br')
    elif si.udFull:
        if frame1 is not None:
            frame1 = draw_text(frame1,labels[0],si.labelPos if si.labelPos else 'tl')
        if frame2 is not None:
            frame2 = draw_text(frame2,labels[1],si.labelPos if si.labelPos else 'bl')
    return frame1, frame2

def drawLine(frame,x=-1,y=-1):
    lineWidthR = 0.002
    lineWidth = min(int(lineWidthR * frame.shape[0] * 2),int(lineWidthR * frame.shape[1]))
    if x >= 0:
        frame[:,x-lineWidth:x+lineWidth,:] = 255
    elif y >= 0:
        frame[y-lineWidth:y+lineWidth,:,:] = 255

def renderShot(shot,r,w,si,labels):
    t = 0 #default
    lp = 0.5
    lt = 3
    fr = 1
    b = False
    s = False

    cmds = shot.split(',')
    if len(cmds) == 0:
        print('Invalid shot script: {}.'.format(shot))
        exit(1)
    for cmd in cmds[1:]:
        if not cmd:
            print('Invalid shot script: {}.'.format(shot))
            exit(1)
        if cmd[0] == 't':
            t = float(cmd[1:])
        elif cmd[0] == 'l':
            paras = cmd[1:].split(':')
            if len(paras) < 2:
                print('Invalid shot script: {}.'.format(shot))
                exit(1)
            lp = float(paras[0])
            lt = float(paras[1])
            if lp < 0 or lp > 1:
                print('Invalid line position: {}. The line position is overwritten by default 0.5.'.format(lp))
        elif cmd[0] == 'r':
            fr = min(16,max(1,int(1/float(cmd[1:]))))
        elif cmd[0] == 'b':
            b = True
        elif cmd[0] == 's':
            s = True

    if cmds[0] == 's1' or cmds[0] == 's2':
        t = 2 if not t else t
        v = True if cmds[0] == 's2' else False
        for i in range(int(r.fps * t)):
            frame1,frame2 = r.getNextFramePairs()
            if frame1 is None or frame2 is None:
                break;
            if v:
                frame1,frame2 = addLabel(si,labels,None,frame2)
                w.frame[:r.h,:r.w,:] = frame2
            else:
                frame1,frame2 = addLabel(si,labels,frame1)
                w.frame[:r.h,:r.w,:] = frame1
            if fr != 1 and args.print_speed:
                w.frame = draw_text(w.frame,'1/{}倍速'.format(fr),'bm',(255,255,0))
            for j in range(int(fr)):
                w.write()
    elif cmds[0] == 's1s2' or cmds[0] == 's12s':
        t = 5 if not t else t
        v = True if cmds[0] == 's12s' else False
        for i in range(int(r.fps * t)):
            frame1,frame2 = r.getNextFramePairs()
            if frame1 is None or frame2 is None:
                break;
            frame1,frame2 = addLabel(si,labels,frame1,frame2)
            w.frame[:r.h,:r.w,:] = frame1
            if v:
                w.frame[r.h:r.h*2,:r.w,:] = frame2
                drawLine(w.frame,-1,r.h)
            else:
                w.frame[:r.h,r.w:r.w*2,:] = frame2
                drawLine(w.frame,r.w,-1)
            if fr != 1 and args.print_speed:
                w.frame = draw_text(w.frame,'1/{}倍速'.format(fr),'bm',(255,255,0))
            for j in range(int(fr)):
                w.write()
    elif cmds[0] == 's1vs2' or cmds[0] == 's1v2s':
        t = 5 if not t else t
        v = True if cmds[0] == 's1v2s' else False
        linePos = int(lp * r.w)
        for i in range(int(r.fps * t)):
            frame1,frame2 = r.getNextFramePairs()
            if frame1 is None or frame2 is None:
                break;
            frame1,frame2 = addLabel(si,labels,frame1,frame2)
            if v:
                w.frame[:linePos,:r.w,:] = frame1[:linePos,:r.w,:]
                w.frame[linePos:r.h,:r.w,:] = frame2[linePos:r.h,:r.w,:]
                drawLine(w.frame,-1,linePos)
            else:
                w.frame[:r.h,:linePos,:] = frame1[:r.h,:linePos,:]
                w.frame[:r.h,linePos:r.w,:] = frame2[:r.h,linePos:r.w,:]
                drawLine(w.frame,linePos,-1)
            if fr != 1 and args.print_speed:
                w.frame = draw_text(w.frame,'1/{}倍速'.format(fr),'bm',(255,255,0))
            for j in range(int(fr)):
                w.write()
    elif cmds[0] == 's2vs1':
        t = 5 if not t else t
        v = True if cmds[0] == 's1v2s' else False
        linePos = int(lp * r.w)
        for i in range(int(r.fps * t)):
            frame1,frame2 = r.getNextFramePairs()
            if frame1 is None or frame2 is None:
                break;
            frame2,frame1 = addLabel(si,labels,frame1,frame2)
            if v:
                w.frame[:linePos,:r.w,:] = frame1[:linePos,:r.w,:]
                w.frame[linePos:r.h,:r.w,:] = frame2[linePos:r.h,:r.w,:]
                drawLine(w.frame,-1,linePos)
            else:
                w.frame[:r.h,:linePos,:] = frame1[:r.h,:linePos,:]
                w.frame[:r.h,linePos:r.w,:] = frame2[:r.h,linePos:r.w,:]
                drawLine(w.frame,linePos,-1)
            if fr != 1 and args.print_speed:
                w.frame = draw_text(w.frame,'1/{}倍速'.format(fr),'bm',(255,255,0))
            for j in range(int(fr)):
                w.write()
    elif cmds[0].find('w') >= 0:
        swipeTime = 1
        t = (lt + swipeTime + 0.1) if not t else t
        d = r.h if cmds[0].find('2s') > 0 else r.w
        v = True if cmds[0].find('2s') > 0 else False
        if s:
            ss = int(np.ceil(r.fps * swipeTime * (1 - lp))) if b else int(np.ceil(r.fps * swipeTime * lp))
        else:
            ss = 0
        staticLinePos = int(lp * d)
        staticBeginFrame = -1
        if cmds[0].find('i') >= 0:
            frame1,frame2 = r.getNextFramePairs()
            if frame1 is None or frame2 is None:
                return
            frame1,frame2 = addLabel(si,labels,frame1,frame2)
        for i in range(ss, int(r.fps * t + ss)):
            if cmds[0].find('v') >= 0:
                frame1,frame2 = r.getNextFramePairs()
                if frame1 is None or frame2 is None:
                    break;
                frame1,frame2 = addLabel(si,labels,frame1,frame2)
            if cmds[0][-1] == 'r':
                linePos = int(d - i * d / (r.fps * swipeTime))
                if linePos < staticLinePos:
                    if staticBeginFrame < 0:
                        staticBeginFrame = i
                    if staticBeginFrame >= 0 and i < staticBeginFrame + r.fps * lt:
                        linePos = staticLinePos
                    elif b:
                        linePos = int(staticLinePos + (i - staticBeginFrame - r.fps * lt) * d / (r.fps * swipeTime))
                    else:
                        linePos = int(staticLinePos - (i - staticBeginFrame - r.fps * lt) * d / (r.fps * swipeTime))
            else:
                linePos = int(i * d / (r.fps * swipeTime))
                if linePos > staticLinePos:
                    if staticBeginFrame < 0:
                        staticBeginFrame = i
                    if staticBeginFrame >= 0 and i < staticBeginFrame + r.fps * lt:
                        linePos = staticLinePos
                    elif b:
                        linePos = int(staticLinePos - (i - staticBeginFrame - r.fps * lt) * d / (r.fps * swipeTime))
                    else:
                        linePos = int(staticLinePos + (i - staticBeginFrame - r.fps * lt) * d / (r.fps * swipeTime))
            if linePos < 0:
                linePos = 0
            if linePos >=d:
                linePos = d-1;
            if v:
                w.frame[:linePos,:r.w,:] = frame2[:linePos,:r.w,:]
                w.frame[linePos:r.h,:r.w,:] = frame1[linePos:r.h,:r.w,:]
                drawLine(w.frame,-1,linePos)
            else:
                w.frame[:r.h,:linePos,:] = frame2[:r.h,:linePos,:]
                w.frame[:r.h,linePos:r.w,:] = frame1[:r.h,linePos:r.w,:]
                drawLine(w.frame,linePos,-1)
            if fr != 1 and args.print_speed:
                w.frame = draw_text(w.frame,'1/{}倍速'.format(fr),'bm',(255,255,0))
            for j in range(int(fr)):
                w.write()

args = get_arguments()
if not os.path.exists(args.output):
    os.mkdir(args.output)
args.output_subdir = os.path.join(args.output, 'separate')
if not os.path.exists(args.output_subdir):
    os.mkdir(args.output_subdir)
if args.timespan:
    args.timespan = get_timespan(args.timespan)
if not args.framebias:
    args.framebias = 0

resizeRatio1 = 1
resizeRatio2 = 1
padding1=[0,0]
padding2=[0,0]
if args.samesize or args.compare:
    size1 = get_resolution(args.input1)
    size2 = get_resolution(args.input2)
    if size1[0] < size2[0]:
        if size2[0] / size1[0] <= size2[1] / size1[1]:
            resizeRatio1 = size2[0] / size1[0]
            padding1[1] = size2[1]-int(size1[1]*resizeRatio1);
        else:
            resizeRatio1 = size2[1] / size1[1]
            padding1[0] = size2[0]-int(size1[0]*resizeRatio1);
    elif size2[0] < size1[0]:
        if size1[0] / size2[0] <= size1[1] / size2[1]:
            resizeRatio2 = size1[0] / size2[0]
            padding2[1] = size1[1]-int(size2[1]*resizeRatio2);
        else:
            resizeRatio2 = size1[1] / size2[1]
            padding2[0] = size1[0]-int(size2[0]*resizeRatio2);

bitrateRatio1 = 1
if args.bitrate1 and float(args.bitrate1) != 1:
    bitrateRatio1 = float(args.bitrate1)
if args.samefps:
    frameRate2 = get_framerate(args.input2)
elif args.frameRate > 0:
    frameRate2 = args.frameRate
else:
    frameRate2 = 0
if args.compare:
    args.samesize = True

date = datetime.datetime.now()
date = '%02d-%02d' % (date.month, date.day)
input1_name = os.path.basename(args.input1).split('.')[0]
input2_name = os.path.basename(args.input2).split('.')[0]
outputBase1 = os.path.join(args.output_subdir, '%s_%s_%s' % (date, input1_name, args.suffix))
outputBase2 = os.path.join(args.output_subdir, '%s_%s_%s' % (date, input2_name, args.suffix))

if args.bbox:
    bbox = get_bbox(args.bbox)
    if args.time:
        status,output1 = cut_crop_video(args.input1,outputBase1+'_out1.png',bbox,alter_time(args.time,args.framebias),"",resizeRatio1,bitrateRatio1,frameRate2)
        status,output2 = cut_crop_video(args.input2,outputBase2+'_out2.png',bbox,args.time,"",resizeRatio2,1,0)
    elif args.timespan:
        status,output1 = cut_crop_video(args.input1,outputBase1+'_out1.mp4',bbox,alter_time(args.timespan[0],args.framebias),alter_time(args.timespan[1],args.framebias),resizeRatio1,bitrateRatio1,frameRate2)
        status,output2 = cut_crop_video(args.input2,outputBase2+'_out2.mp4',bbox,args.timespan[0],args.timespan[1],resizeRatio2,1,frameRate2)
    else:
        status,output1 = crop_video(args.input1,outputBase1+'_out1'+os.path.splitext(args.input1)[1],bbox,resizeRatio1,bitrateRatio1,frameRate2)
        status,output2 = crop_video(args.input2,outputBase2+'_out2'+os.path.splitext(args.input2)[1],bbox,resizeRatio2,1,0)
else:
    if args.timespan:
        status,output1 = cut_video(args.input1,outputBase1+'_out1.mp4',alter_time(args.timespan[0],args.framebias),alter_time(args.timespan[1],args.framebias),resizeRatio1,bitrateRatio1,frameRate2,padding1)
        status,output2 = cut_video(args.input2,outputBase2+'_out2.mp4',args.timespan[0],args.timespan[1],resizeRatio2,1,0,padding2)
    elif args.time:
        status,output1 = cut_video(args.input1,outputBase1+'_out1.png',alter_time(args.time,args.framebias),"",resizeRatio1,bitrateRatio1,frameRate2,padding1)
        status,output2 = cut_video(args.input2,outputBase2+'_out2.png',args.time,"",resizeRatio2,1,0,padding2)
    elif args.samesize:
        status,output1 = resize_video(args.input1,outputBase1+'_rs1'+os.path.splitext(args.input1)[1],resizeRatio1,bitrateRatio1,frameRate2,padding1)
        status,output2 = resize_video(args.input2,outputBase2+'_rs2'+os.path.splitext(args.input2)[1],resizeRatio2,1,0,padding2)
    else:
        output1 = args.input1
        output2 = args.input2

if args.compare:
    shots = args.compare.split('-')
    labels = args.label.split('-') if args.label is not None else None
    si = ScriptInterpreter(shots,args.labelposition if args.labelposition else None)
    r = Reader(output1,output2)
    save_name = os.path.join(args.output, date + '_' + input1_name+'_'+input2_name+'_%s.mp4'%args.suffix)
    print(save_name)
    
    w = Writer(r,si,save_name,args.bitrateOutput if args.bitrateOutput else None,crf=args.crf)
    for shot in shots:
        renderShot(shot,r,w,si,labels)
