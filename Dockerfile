# 指定基础映像
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# 维护者信息
LABEL maintainer="960099622@qq.com"

# 设置工作目录
WORKDIR /app

# RUN pip install torch==2.5.1+cu124 torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu124/

# 复制依赖文件并安装
COPY requirements_docker.txt .

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com

RUN pip install --no-cache-dir -r requirements_docker.txt
# 复制应用代码到容器
COPY . .

# 暴露应用端口
EXPOSE 18085

# 容器启动命令
CMD ["python", "app.py"]