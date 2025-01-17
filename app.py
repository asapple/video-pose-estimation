from flask import Flask, request, jsonify
import threading
import os
import subprocess
import time
import signal

app = Flask(__name__)
background_type="origin"
# 确保output目录存在
if not os.path.exists('output'):
    os.makedirs('output')

# 存储正在运行的进程（按设备ID分类）
running_processes = {}

# RTSP流处理与关键点检测推理函数
def process_video(device_id, rtsp_url, background_type):
    timestamp = int(time.time())
    output_file = f'output/{device_id}_{timestamp}.mp4'

    try:
        process = subprocess.Popen(["python", "pose-estimate.py", "--source", rtsp_url, "--device", 'cpu', "--output", output_file, "--background_type", background_type])
        running_processes[device_id] = process  # 存储进程对象
        print(f"处理视频流，设备ID: {device_id}, 输出文件: {output_file}")
        process.wait()  # 等待进程完成
    except subprocess.CalledProcessError as e:
        print(f"调用pose-estimate.py时出错: {e}")
    finally:
        # 处理完成后移除进程对象
        running_processes.pop(device_id, None)


# 后端接口：接收RTSP流地址并开始处理
@app.route('/realtime/play', methods=['GET'])
def play_video():
    device_id = request.args.get('device_id')
    rtsp_url = request.args.get('rtsp')

    if not device_id or not rtsp_url:
        return jsonify({"error": "缺少device_id或rtsp参数"}), 400

    # 创建新线程来处理视频流
    threading.Thread(target=process_video, args=(device_id, rtsp_url, background_type)).start()

    return jsonify({"message": f"正在处理设备ID: {device_id}, 请稍等..."}), 200


# 接口：背景类型设置
@app.route('/background/<type>', methods=['GET'])
def set_background(type):
    global background_type
    if type not in ['origin', 'hidden']:
        return jsonify({"error": "无效的背景类型"}), 400

    # 获取背景类型并传递给处理视频函数
    background_type = type
    return jsonify({"message": f"背景类型已设置为 {background_type}"}), 200


# 接口：调用本地摄像头
@app.route('/realtime/camera', methods=['GET'])
def camera_video():
    try:
        # 使用subprocess.Popen来启动进程并保存进程对象
        process = subprocess.Popen(["python", "pose-estimate.py", "--source", "0", "--view-img", "--background_type", background_type])

        # 保存进程对象到running_process字典，便于后续终止
        running_processes['camera'] = process

        return jsonify({"message": "正在使用本地摄像头处理视频流..."}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"调用本地摄像头时出错: {e}"}), 500

# 接口：本地测试视频
@app.route('/realtime/video', methods=['GET'])
def video_processing():
    try:
        # 使用subprocess.Popen来启动进程并保存进程对象
        process = subprocess.Popen(["python", "pose-estimate.py", "--source", "football1.mp4", "--device", "cpu"])

        # 保存进程对象到running_processes字典，便于后续终止
        running_processes['video'] = process

        return jsonify({"message": "正在处理视频文件: football1.mp4..."}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"调用视频文件处理时出错: {e}"}), 500

# 接口：停止测试视频处理
@app.route('/realtime/video/stop', methods=['GET'])
def stop_video_processing():
    if 'video' not in running_processes:
        return jsonify({"error": "没有正在运行的视频处理进程"}), 400

    # 获取正在运行的进程并终止它
    process = running_processes['video']
    process.terminate()  # 发送终止信号
    running_processes.pop('video', None)  # 移除进程记录

    return jsonify({"message": "视频处理已停止"}), 200


# 接口：停止本地摄像头处理
@app.route('/realtime/camera/stop', methods=['GET'])
def stop_camera_video():
    if 'camera' not in running_processes:
        return jsonify({"error": "没有正在运行的本地摄像头进程"}), 400

    # 获取正在运行的进程并终止它
    process = running_processes['camera']
    process.terminate()  # 发送终止信号
    running_processes.pop('camera', None)  # 移除进程记录

    return jsonify({"message": "本地摄像头处理已停止"}), 200


# 接口：停止RTSP视频流处理
@app.route('/realtime/stop', methods=['GET'])
def stop_video():
    device_id = request.args.get('device_id')

    if not device_id:
        return jsonify({"error": "缺少device_id参数"}), 400

    if device_id not in running_processes:
        return jsonify({"error": f"设备ID {device_id} 的视频流处理未启动"}), 400

    # 获取正在运行的进程并终止它
    process = running_processes[device_id]
    process.terminate()  # 发送终止信号
    running_processes.pop(device_id, None)  # 移除进程记录

    return jsonify({"message": f"设备ID {device_id} 的视频流处理已停止"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
