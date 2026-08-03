# Layout Classifier — 文档版式分类平台

基于 OpenAI CLIP ViT-B/32 视觉编码器的文档版式分类系统，提供数据集管理 → 模型训练 → 推理服务部署的完整工作流。

## 项目结构

```
clip/
├── models/                            # CLIP ViT-B/32 预训练权重（HuggingFace openai/clip-vit-base-patch32）
│   ├── config.json
│   ├── pytorch_model.bin              # ~580 MB，需单独下载
│   └── ...
│
└── vit/                               # 主工程
    ├── configs/
    │   ├── train_config.yaml          # 训练超参数 & 类别定义
    │   └── server_config.example.yaml # 服务配置模板（复制为 server_config.yaml 后填写）
    ├── src/                           # 核心模型库
    │   ├── data/dataset.py            # 数据集加载 / 数据增强
    │   ├── models/classifier.py       # CLIP ViT + 分类头
    │   ├── trainers/trainer.py        # 训练循环 / Early Stop / Checkpoint
    │   └── inference/predictor.py     # 推理引擎
    ├── services/                      # 独立微服务
    │   ├── training/                  # 训练服务（GPU 容器）
    │   ├── inference/                 # 推理服务（GPU 容器）
    │   └── docker-compose.yml         # 三服务编排
    ├── server/                        # Backend Web 服务（项目 / 数据集管理）
    ├── frontend/                      # React + Ant Design 管理界面
    └── scripts/
        ├── train.py                   # CLI 训练入口（独立使用）
        ├── serve.py                   # CLI 推理入口（独立使用）
        └── webapp.py                  # Web 平台启动入口
```

## 架构

```
┌─────────────────────────┐
│   Frontend (React)      │  管理界面：项目管理 / 数据上传 / 训练监控 / 推理验证
└───────────┬─────────────┘
            │ HTTP
┌───────────┴─────────────┐
│  Backend Web Service    │  职责：CRUD、S3 图片管理、编排训练/推理服务
│  (server/)              │  不加载模型，纯 HTTP 调用下游服务
└─────┬──────────┬────────┘
      │ HTTP     │ HTTP (Base64)
      ▼          ▼
┌──────────┐ ┌──────────────┐
│ Training │ │  Inference   │
│ Service  │ │  Service     │
│ 包装     │ │  包装        │
│ trainer  │ │  predictor   │
└──────────┘ └──────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# Python 环境（推荐 conda）
conda create -n ocr python=3.12 -y
conda activate ocr
pip install -r requirements.txt

# CLIP 模型权重
# 下载 openai/clip-vit-base-patch32 到 models/ 目录
```

### 2. 配置文件

```bash
# 复制示例配置并填写真实信息
cp configs/server_config.example.yaml configs/server_config.yaml
# 编辑 server_config.yaml，填写 MySQL 连接信息、S3 凭证等
```

### 3. 初始化数据库

```sql
CREATE DATABASE IF NOT EXISTS layout_classifier CHARACTER SET utf8mb4;
-- 启动服务后自动建表
```

### 4. 启动 Web 平台

```bash
# 开发模式
# Terminal 1: 训练服务
uvicorn services.training.app:app --host 0.0.0.0 --port 8001

# Terminal 2: 推理服务
uvicorn services.inference.app:app --host 0.0.0.0 --port 8002

# Terminal 3: Backend Web + 前端
python scripts/webapp.py

# 浏览器打开 http://localhost:8000
# API 文档 http://localhost:8000/docs
```

### 5. 容器化部署

```bash
cd services
docker-compose up -d
# Backend :8000 | Training :8001 | Inference :8002
```

---

## 独立 CLI 使用（不依赖 Web 平台）

### 训练

```bash
# 准备数据（按类别分文件夹）
# data/raw/身份证/  data/raw/发票/  data/raw/...

# 编辑 configs/train_config.yaml
python scripts/train.py --config configs/train_config.yaml
```

训练流程：
1. `src/data/dataset.py` 按文件夹读取图片，自动标注类别
2. 数据按 80/20 划分为训练/验证集
3. `src/models/classifier.py` 冻结 CLIP ViT 编码器，追加 Dropout + Linear 分类头
4. `src/trainers/trainer.py` 执行训练循环、Cosine 学习率调度、Early Stop
5. 输出 `output/best_model.pth` 和 `output/training_history.json`

### 推理

```bash
# 启动推理 API 服务
python scripts/serve.py --checkpoint output/best_model.pth

# 调用 API
curl -X POST http://localhost:8000/predict \
  -F "file=@test.jpg"
```

### 环境验证

```bash
python scripts/verify.py
# 检查 GPU 可用性、模型加载、数据集加载
```

---

## 训练配置说明 (`configs/train_config.yaml`)

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `data.categories` | 版式类别，每个类别对应一个数据文件夹 | - |
| `model.freeze_encoder` | 冻结 CLIP 编码器，只训练分类头 | true |
| `model.pool` | 特征提取方式：cls / mean | cls |
| `training.batch_size` | 批大小 | 32 |
| `training.epochs` | 最大训练轮数 | 50 |
| `training.lr` | 学习率 | 0.001 |
| `training.lr_scheduler` | 学习率调度器：cosine / step | cosine |
| `training.early_stop_patience` | 验证精度不再提升的容忍轮数 | 10 |
| `training.class_balance` | 类别均衡策略：none / weighted_loss / oversample | weighted_loss |

## 类别均衡策略

| 策略 | 原理 |
|------|------|
| `none` | 不做处理 |
| `weighted_loss` | 反频率权重注入 CrossEntropyLoss，少数类权重更大 |
| `oversample` | WeightedRandomSampler 等概率采样每个类别 |

## 数据准备格式

```
data/raw/
├── 身份证/
│   ├── img001.jpg
│   └── img002.png
├── 发票/
│   └── ...
└── 合同/
    └── ...
```

图片支持格式：`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp`
