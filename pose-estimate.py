import cv2
import time
import torch
import argparse
import numpy as np
import os
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt
from download_image import get_or_download_image  # 导入下载图片功能
import subprocess
import matplotlib as plt
import logging

# 初始化日志配置
logging.basicConfig(
    filename='video_pose_detect.log',  # 日志文件名
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="football1.mp4", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True, output="output/video_output.mp4", background_type='origin', device_id="default"):
    frame_count = 0  # count no of frames
    total_fps = 0  # count total fps
    time_list = []  # list to store time
    fps_list = []  # list to store fps

    device = select_device(device)  # select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  # Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))  # pass video to VideoCapture object
    else:
        cap = cv2.VideoCapture(source)  # pass video to VideoCapture object

    if not cap.isOpened():  # check if video capture opened successfully
        logging.info('Error while trying to read video. Please check path again')
        raise SystemExit()

    frame_width = int(cap.get(3))  # get video frame width
    frame_height = int(cap.get(4))  # get video frame height

    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]  # init video writer
    resize_height, resize_width = vid_write_image.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))

    # RTMP URL for FFmpeg
    rtsp_url = None
    if background_type == 'origin':
        rtsp_url = f"rtmp://8.149.243.30/inference/{device_id}"  
    else:
        rtsp_url = f"rtmp://8.149.243.30/inference/{device_id}_hidden"
    logging.info(f"开始推流：{rtsp_url}")  
    # 初始化FFmpeg推流进程
#     command = [
#     'ffmpeg',
#     '-y',
#     '-f', 'rawvideo',
#     '-vcodec', 'rawvideo',
#     '-pix_fmt', 'bgr24',
#     '-s', f'{resize_width}x{resize_height}',
#     '-r', str(fps),
#     '-i', '-',
#     '-c:v', 'libx264',
#     '-pix_fmt', 'yuv420p',
#     '-preset', 'ultrafast',
#     '-f', 'flv',
#     rtsp_url
# ]
    command = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{resize_width}x{resize_height}', 
    '-r', '15',       # 输入帧率提升至全帧率30
    '-i', '-',
    '-vf', 'format=yuv420p,fps=15',  # 输出帧率同步
    '-c:v', 'h264_nvenc',
    '-rc', 'vbr',          # 变码率模式
    '-b:v', '64k',         # 平均码率64Kbps
    '-maxrate', '96k',     # 最高瞬时码率（变码型兼容）
    '-minrate', '32k',     # 最低瞬时码率
    '-bufsize', '128k',    # 缓冲区=2倍平均码率
    '-g', '200',           # 关键帧间隔200帧
    '-tune', 'll',         # 低延迟模式
    '-pix_fmt', 'yuv420p',
    '-f', 'flv',
    rtsp_url
]
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)
    logging.info(f"开始推流:{rtsp_url}")

    # Load background image if type is 'hidden'
    background = None
    if background_type == 'hidden':
        background_path = get_or_download_image(device_id)  # Get image path using the new function
        background = cv2.imread(background_path)  # Read the background image

        if background is None:
            logging.info(f"Error: Unable to load background image from {background_path}")
            return
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  # Convert to RGB as background
        background = letterbox(background, (frame_width), stride=64, auto=True)[0]
        background = background.copy()
        background = transforms.ToTensor()(background)
        background = torch.tensor(np.array([background.numpy()]))
        background = background.to(device)  # convert image data to device
        background = background.float()  # convert image to float precision (cpu)

    while(cap.isOpened()):  # loop until cap opened or video not complete
        logging.info("Frame {} Processing".format(frame_count + 1))
        ret, frame = cap.read()  # get frame and success from video capture
        if ret:  # if success is true, means frame exists
            orig_image = frame  # store frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # convert frame to RGB
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))

            image = image.to(device)  # convert image data to device
            image = image.float()  # convert image to float precision (cpu)
            start_time = time.time()  # start time for fps calculation
            # frame_queue.put(image)
            with torch.no_grad():  # get predictions
                output_data, _ = model(image)
            # output_data = result_queue.get()
            output_data = non_max_suppression_kpt(output_data,  # Apply non max suppression
                                                  0.25,  # Conf. Threshold.
                                                  0.65,  # IoU Threshold.
                                                  nc=model.yaml['nc'],  # Number of classes.
                                                  nkpt=model.yaml['nkpt'],  # Number of keypoints.
                                                  kpt_label=True)

            output = output_to_keypoint(output_data)
            # Use background if type is 'hidden', otherwise use original frame
            if background_type == 'hidden':
                im0 = background[0].permute(1, 2, 0) * 255  # 使用背景图片
            else:
                im0 = image[0].permute(1, 2, 0) * 255  # 使用原始帧

            im0 = im0.cpu().numpy().astype(np.uint8)
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for i, pose in enumerate(output_data):  # detections per image

                if len(output_data):  # check if no pose
                    for c in pose[:, 5].unique():  # Print results
                        n = (pose[:, 5] == c).sum()  # detections per class
                        logging.info("No of Objects in Current Frame : {}".format(n))

                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):  # loop over poses for drawing on frame
                        c = int(cls)  # integer class
                        kpts = pose[det_index, 6:]
                        hide_conf = True  # 使得置信度不显示
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                         line_thickness=0, kpt_label=True, kpts=kpts, steps=3,
                                         orig_shape=im0.shape[:2])

            end_time = time.time()  # Calculation for FPS
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            fps_list.append(total_fps)  # append FPS in list
            time_list.append(end_time - start_time)  # append time in list

            # Stream results
            if view_img:
                cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                cv2.waitKey(1)  # 1 millisecond

            # out.write(im0)  # writing the video frame
            # 将帧写入FFmpeg管道
            try:
                ffmpeg_process.stdin.write(im0.tobytes())
            except Exception as e:
                logging.error(f"Error writing to FFmpeg: {e}")
                break
        else:
            logging.info("未获取到流结果，重新连接...")
            if source.isnumeric():
                cap = cv2.VideoCapture(int(source))  # pass video to VideoCapture object
            else:
                cap = cv2.VideoCapture(source)  # pass video to VideoCapture object
            if not cap.isOpened():  # check if video capture opened successfully
                logging.info('Error while trying to read video. Please check path again')
            

    cap.release()
    avg_fps = total_fps / frame_count
    logging.info(f"Average FPS: {avg_fps:.3f}")

    # plot the comparison graph
    # plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)
    if ffmpeg_process.poll() is None:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
    logging.info("Streaming finished")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')  # video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')  # device arguments
    parser.add_argument('--view-img', action='store_true', help='display results')  # display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')  # save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')  # box line thickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # box hide labels
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # box hide conf
    parser.add_argument('--output', type=str, default='output/video_output.mp4', help='output video file path')  # output file path
    parser.add_argument('--background_type', type=str, default='origin', choices=['origin', 'hidden'], help='background type: origin or hidden')  # background type
    parser.add_argument('--device_id', type=str, required=True, help="Device ID for background image")  # device id
    opt = parser.parse_args()
    return opt


#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list, fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list, 'b', label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")


#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    # strip_optimizer(opt.device, opt.poseweights)
    main(opt)
