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

### 6. 离线服务器部署（内网无网络环境）

镜像构建完成后，所有需要联网下载的内容（Python 依赖、CJK 字体、CLIP 模型权重、前端 node_modules 与初始 dist）都已打入镜像，内网服务器无需任何外网访问。

**构建机（联网，一次性）：**

```bash
cd vit
docker compose build            # 构建 backend / training / inference 三个镜像
docker save backend training inference -o images.tar
# 预期 6~7GB（含 CUDA/cuDNN 层），拷到离线服务器后：
```

**服务器（离线）：**

```bash
docker load < images.tar
# 项目目录只需拷贝到服务器（代码经 bind mount 挂载，改代码后重启容器即生效）：
#   vit/ 目录（至少包含 configs/server_config.yaml）
#   可选：vit/frontend/dist 等——容器启动时会自动用镜像内 node_modules 离线重建 dist
cd vit/services && docker compose up -d
```

**服务器前置条件：**
- NVIDIA GPU 驱动 + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（GPU 训练/推理必需）
- MySQL 数据库、S3/MinIO 对象存储（内网可达；训练数据经 S3 presigned URL 下载）
- `vit/configs/server_config.yaml`：`mysql.*` 与 `s3.*` 必须填**服务器内网可达地址**（容器内 127.0.0.1 指向容器自身，需用内网 IP 或容器网络别名）

**验证：**

```bash
curl http://<server>:8000/api/health    # backend
curl http://<server>:8001/api/health    # training（含 cuda_available）
curl http://<server>:8002/api/health    # inference
# 浏览器访问 http://<server>:8000 应看到管理界面（首次启动日志含 "Frontend build OK"）
```

**日常更新（无需重建镜像）：**
- 改后端代码（`server/`、`src/`、`services/*/app.py`）→ 重启对应容器即可（Python 无编译）
- 改前端源码（`frontend/`）→ 重启 backend 容器，启动时自动 `npm run build` 生成新 dist
- ⚠️ 新增依赖需外网重建镜像：前端改 `package.json`（node_modules 快照在镜像内）、后端引用镜像中没有的新 Python 包

---

## 端到端流程（训练 → 推理）

### 方式一：Web 平台（推荐）

```
① 新建项目        前端 Dashboard → 新建项目（名称 / 描述 / 模型路径）
② 上传数据集      Dataset 页 → 选择类别 → 上传图片 / PDF / OFD 文档
                  （文档自动按页拆分为独立训练样本；支持增删改类别、8:2 划分 train/val）
③ 启动训练        Training 页 → 配置超参数 → Start Training
                  （训练完成后自动执行温度校准 + 阈值生成，写入 checkpoint）
④ 查看模型        Models 页 → 训练产物列表（val_acc / 创建时间）
⑤ 推理验证        Inference 页 → 上传图片或 PDF/OFD → 逐页显示预测结果
                  （置信度不足的页/图显示"其他"+ 原因，阈值自动生效）
⑥ 部署推理服务    Services 页 → 按示例用 scripts/serve.py 或推理微服务部署
```

### 方式二：CLI（不依赖 Web 平台）

```bash
# ① 准备数据（每类别一个文件夹）
#     data/raw/身份证/  data/raw/发票/  data/raw/合同/

# ② 训练（自动生成校准阈值）
python scripts/train.py --config configs/train_config.yaml
# 输出 output/best_model.pth（内嵌 calibration）+ calibration_report.json

# ③ 启动推理服务（阈值自动从 checkpoint 加载）
python scripts/serve.py --checkpoint output/best_model.pth

# ④ 调用推理
curl -X POST http://localhost:8000/predict -F "file=@test.jpg"
```

### 阈值在训练→推理中的流转

```
训练阶段（Trainer.calibrate）               推理阶段（LayoutPredictor）
────────────────────────────             ────────────────────────────
1. 验证集收集 logits                       1. logits ÷ T（温度缩放）
2. 拟合温度 T（最小化 NLL）                 2. softmax → 校准后概率
3. 正确样本 P5 分位数 × 0.9                 3. 双阈值判定：
   → confidence_threshold                    · max < 阈值 → 其他(low_confidence)
   → margin_threshold                        · top1-top2 < 阈值 → 其他(ambiguous)
4. 写入 checkpoint["calibration"]     ────▶ 4. 加载 checkpoint 时自动读取，零配置生效
5. 输出 calibration_report.json             5. 支持请求/配置级临时覆盖
```

**阈值覆盖优先级**（高 → 低）：

| 层级 | CLI | Web 平台 |
|------|-----|----------|
| ① 单次调用 | `--confidence-threshold` 等启动参数 | 推理请求 Form 字段（图片与文档接口均支持） |
| ② 后端全局 | `CONFIDENCE_THRESHOLD` 等环境变量 | `server_config.yaml` 的 `inference` 段 |
| ③ checkpoint 内置 | 训练时自动生成（推荐，无需配置） | 同左 |
| ④ 代码默认 | 0.6 / 0.1 / T=1 | 同左 |

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
5. 训练完成后自动执行**温度校准 + 置信度阈值生成**（`Trainer.calibrate()`，见下文「置信度拒绝机制」）

训练输出（`output/` 目录）：

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最佳 checkpoint，内含 `calibration` 字段（温度 T、双阈值） |
| `training_history.json` | 每 epoch 的 loss / acc / lr |
| `calibration_report.json` | 校准报告：置信度/间隔分位数表 + 覆盖率-精度曲线（用于阈值调优） |

### 推理

```bash
# 启动推理 API 服务（阈值默认取自 checkpoint 内嵌校准参数，零配置生效）
python scripts/serve.py --checkpoint output/best_model.pth

# 可选：临时覆盖阈值（不传则用 checkpoint 内置值）
python scripts/serve.py --checkpoint output/best_model.pth \
  --confidence-threshold 0.7 --margin-threshold 0.15 --temperature 1.2

# 调用 API
curl -X POST http://localhost:8000/predict \
  -F "file=@test.jpg"
```

响应示例（置信度不足时归为"其他"）：

```json
{ "category": "其他",
  "index": -1,
  "confidence": 0.55,
  "probabilities": { "身份证": 0.55, "发票": 0.44, "合同": 0.01 },
  "rejected": true,
  "reason": "ambiguous",
  "original_category": "身份证",
  "original_index": 0 }
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

### PDF / OFD 文档支持（Web 平台）

Web 平台额外支持上传 `.pdf` 和 `.ofd` 文档：

- **训练集上传**：文档按页渲染为 PNG（PDF 150 DPI，长边超 2000px 自动缩小），每页作为独立训练样本归入所选类别，文件名自动带页码后缀（如 `合同_p001.png`）。
- **推理上传**：`/predict` 与 `/predict/top-k` 接口接收文档后逐页推理，返回逐页结果列表：
  ```json
  { "filename": "合同.pdf", "page_count": 3,
    "results": [ {"page": 1, "category": "合同", "confidence": 0.98, "probabilities": {...}}, ... ] }
  ```
- 限制：单文档最多转换 50 页；加密 PDF 不支持；OFD 文本渲染依赖服务器中文字体（图片类内容不受影响）。

---

## 置信度拒绝机制（归为"其他"）

模型对训练类别之外的文档无法表达"不认识"（softmax 总和恒为 1）。因此训练完成后自动执行**温度校准 + 阈值生成**，推理时置信度不足的输入归为 `其他`：

### 训练时自动生成阈值（`calibration` 配置段）

训练完成后（`Trainer.calibrate()`）自动执行：

1. 加载 `best_model.pth` 跑一遍验证集收集 logits
2. **温度校准**：拟合温度 T（最小化校准后概率的 NLL，`scipy.optimize`），校准后的概率才具有真实语义
3. 用校准后概率统计**正确样本**的分布，生成两个阈值：
   - `confidence_threshold` = top-1 置信度的 P5 分位数 × 0.9（95% 的正确样本不会被误拒）
   - `margin_threshold` = top1-top2 间隔的 P5 分位数 × 0.9
4. 参数**写回 checkpoint**（`best_model.pth` 的 `calibration` 字段），并生成 `calibration_report.json`（分位数表 + 覆盖率/精度曲线，用于人工复核调参）

关键配置（`configs/train_config.yaml`）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `calibration.enabled` | true | 是否自动校准 |
| `calibration.percentile` | 5 | 正确样本分位数（%），越大阈值越严 |
| `calibration.safety_factor` | 0.9 | 分位数乘系数，防验证集乐观偏差 |
| `calibration.confidence_threshold_min/max` | 0.3 / 0.9 | 阈值钳制范围 |

### 推理判定逻辑

```
probs = softmax(logits / T)
若 max(probs) < confidence_threshold      → 归为"其他"（low_confidence）
若 max(probs) - top2(probs) < margin_threshold → 归为"其他"（ambiguous）
否则 → 正常返回 top-1
```

响应新增字段（向后兼容，图片推理结构不变）：

```json
{ "category": "其他",
  "confidence": 0.55,
  "rejected": true,
  "reason": "ambiguous",
  "original_category": "合同",
  "original_index": 2,
  "probabilities": { ... } }
```

参数优先级：**请求参数 > 后端 `server_config.yaml` 的 `inference` 段 > checkpoint 内置 > 代码默认**。旧 checkpoint 无 `calibration` 字段时使用默认值（0.6 / 0.1 / T=1），无需重训即可升级。

### 阈值调优方法

自动生成的阈值适合大部分场景；若觉得误拒/漏拒比例不合适，用 `calibration_report.json` 数据驱动调优：

1. **查看分位数表**：`confidence_percentiles` / `margin_percentiles` 显示正确样本的置信度分布（p1~p90）。若 p5 远高于当前阈值，说明阈值可放宽（误拒少）；若 p5 低于阈值，说明大量正常样本会被拒绝，应调严。
2. **查看覆盖率曲线**：`coverage_curve` 列出各候选阈值下的保留样本比例（coverage）与保留样本精度（accuracy）。业务目标通常是「覆盖率 ≥ 95% 且精度高」，据此选择运行点。
3. **调整方式**（三选一，由易到难）：
   - 不改阈值，先看曲线选运行点，用推理请求参数/`server_config.yaml` 临时覆盖验证效果；
   - 调整 `train_config.yaml` 的 `calibration.percentile`（如 1=只拒最差 1% 的正确样本，10=更保守）或 `safety_factor`，重训自动重新生成；
   - 若手头有"其他类型"的负样本（未训练过的单据、乱拍照片等），可统计负样本的置信度分布，将阈值定在正负样本分布的交界处（更严格、更可靠）。
4. **复核**：每次推理响应都保留 `original_category` 与完整 `probabilities`，即使被拒也可人工判断模型"想选什么"，用于持续评估阈值合理性。
