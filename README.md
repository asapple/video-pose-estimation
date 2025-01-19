# 🚀yolov7-pose-estimation

### 🎉 New Features
- 📊 Added Comparison Graph for FPS & Time
- 💻 How to Run Code in Google Colab
- 🖥️ Supports CPU & GPU
- 🎥 Video/WebCam/External Camera/IP Stream Support

### 🔜 Coming Soon
- 📈 Streamlit Dashboard for Pose-Estimation

### 🚀 Steps to Run Code
- **Google Colab Users**: First, mount the drive:
  ```python
  from google.colab import drive
  drive.mount("/content/drive")
  ```
- **Clone the repository**:
  ```bash
  git clone https://github.com/RizwanMunawar/yolov7-pose-estimation.git
  ```
- **Go to the cloned folder**:
  ```bash
  cd yolov7-pose-estimation
  ```
- **Create a virtual environment** (recommended):
  ```bash
  # Linux
  python3 -m venv psestenv
  source psestenv/bin/activate

  # Windows
  python3 -m venv psestenv
  cd psestenv/Scripts
  activate
  ```
- **Upgrade pip**:
  ```bash
  pip install --upgrade pip
  ```
- **Install requirements**:
  ```bash
  pip install -r requirements.txt
  ```
- **Download YOLOv7 weights** and move to the working directory:
  [yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

- **Run the code**:
  ```bash
  python pose-estimate.py

  # Options:
  python pose-estimate.py --source "your-video.mp4" --device cpu  # For CPU
  python pose-estimate.py --source 0 --view-img  # For Webcam
  python pose-estimate.py --source "rtsp://your-ip" --device 0 --view-img  # For LiveStream
  ```

- Output: The processed video will be saved as **your-file-keypoint.mp4**

### 📊 RESULTS

<table>
  <tr>
    <td>⚽ Football Match</td>
    <td>🏏 Cricket Match</td>
    <td>📈 FPS & Time Comparison</td>
    <td>📡 Live Stream</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/185089411-3f9ae391-ec23-4ca2-aba0-abf3c9991050.png" width=300></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185228806-4ba62e7a-12ef-4965-a44a-6b5ba9a3bf28.png" width=300></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185324844-20ce3d48-f5f5-4a17-8b62-9b51ab02a716.png" width=300></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185587159-6643529c-7840-48d6-ae1d-2d7c27d417ab.png" width=300></td>
  </tr>
</table>

### 🔗 References
- YOLOv7 Repo: https://github.com/WongKinYiu/yolov7
- Ultralytics: https://github.com/ultralytics/yolov5

### 📖 Articles
- [YOLOv7 Training Guide](https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623)
- [Computer Vision Roadmap](https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c)

### How to run
首先运行app.py 运行成功后会在本地5000端口打开接口等待POST请求调用，使用方法如下：
POST http://localhost:5000/realtime/play?device_id=1&rtsp=rtsp://your_rtsp_url 来进行视频流的推送以及推理
POST http://localhost:5000/realtime/stop?device_id=1 停止指定设备的视频流推理
POST http://localhost:5000/realtime/camera 启动本地摄像头进行视频处理
POST http://localhost:5000/realtime/camera/stop 停止本地摄像头处理
POST http://localhost:5000/realtime/video 启动本地测试视频的处理(默认为football1.mp4)
POST http://localhost:5000/realtime/video/stop 停止本地测试视频的处理
以上会app.py接受到相应指令后会调用pose-estimate.py进行推理，推理设备默认CPU可以更改。

设置背景：
POST http://localhost:5000/background/device_id/hidden设置背景为隐去人像，脚本会调用download_image.py中的方法进行下载图片或使用默认空白图
POST http://localhost:5000/background/device_id/origin设置背景为原始帧




