# ============================================================
# 无锡住宅房价预测系统 — Docker 容器化
# 构建: docker build -t house-price-predictor .
# 运行: docker run --rm -p 5000:5000 house-price-predictor
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制全部交付物
COPY . .

# 设置工作目录为 cache (脚本在此目录下运行)
WORKDIR /app/cache

# 设置环境变量(禁用 GUI 进度窗口)
ENV BK_NO_GUI=1

# 默认:启动 Web 演示
EXPOSE 5000
CMD ["python", "web/app.py"]