# 文档信息提取系统

基于 **PaddleOCR-VL-1.6** 的高精度文档信息提取系统，支持 PDF、Word、图片（包括手写体）等多种格式的文字识别和信息提取。

## 特性

- 🚀 **多平台硬件加速**: Mac (Metal) / Windows/Linux (CUDA) / CPU
- 📄 **多格式支持**: PDF、Word、JPG/PNG 等图片格式、手写体
- 🎨 **可视化界面**: 简洁易用的 Web 图形界面
- ⚙️ **灵活配置**: 支持自动硬件检测和参数调整
- 🎯 **高准确率**: 基于 PaddleOCR-VL-1.6 视觉语言模型

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**根据你的平台安装 PaddlePaddle:**

```bash
# Mac (自动支持 M1/M2/M3 Metal 加速)
pip install paddlepaddle>=2.6.0

# Windows/Linux + NVIDIA GPU (以 CUDA 11.7 为例)
pip install paddlepaddle-gpu==2.6.0.post117

# 纯 CPU 版本
pip install paddlepaddle>=2.6.0
```

**安装 poppler (PDF 支持):**
```bash
# Mac
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Windows
# 下载 https://github.com/oschwartz10612/poppler-windows/releases
# 解压并添加 bin 目录到系统 PATH
```

### 2. 运行

```bash
# 启动可视化界面（推荐）
python ocr_extractor.py

# 或者命令行模式
python ocr_extractor.py document.pdf
```

启动后自动打开浏览器访问: **http://127.0.0.1:7860**

## 使用说明

### 可视化界面

**启动方式:**
```bash
# 默认启动
python ocr_extractor.py

# 指定端口
python ocr_extractor.py --web --port 8080

# 局域网访问
python ocr_extractor.py --web --host 0.0.0.0

# 创建公网分享链接
python ocr_extractor.py --web --share
```

**使用步骤:**
1. 进入"系统配置"标签页，点击"初始化系统"
2. 返回"文档提取"标签页
3. 拖拽或选择文件（支持多个文件）
4. 点击"开始提取"
5. 查看文本或 JSON 格式结果

### 命令行模式

```bash
# 处理单个文件
python ocr_extractor.py document.pdf

# 批量处理
python ocr_extractor.py file1.pdf file2.jpg file3.docx

# 使用通配符
python ocr_extractor.py *.pdf

# 自定义配置
python ocr_extractor.py -c config.yaml document.pdf

# 指定输出目录和格式
python ocr_extractor.py -o results/ -f txt document.pdf
```

## 配置说明

编辑 `config.yaml` 自定义参数：

```yaml
# 硬件配置
hardware:
  mode: auto              # auto/cpu/gpu_cuda/gpu_metal
  cpu_threads: 10

# OCR 配置
ocr:
  model_name: "PaddleOCR-VL-1.6"
  use_angle_cls: true     # 自动纠正文字方向
  lang: ch                # 语言: ch/en/french/german/korean/japan

# 图像处理
processing:
  pdf_dpi: 300            # PDF 转图像 DPI
  image_preprocessing: true
  denoise: true
  enhance_contrast: true

# 输出
output:
  format: json            # json/txt
  output_dir: output
  confidence_threshold: 0.5
```

### 模型存储

模型自动下载到 `~/.paddlex/official_models/`。

**如需更改存储位置（例如外置硬盘）:**

```bash
# 删除默认目录
rm -rf ~/.paddlex

# 创建软链接指向自定义位置
ln -s /Volumes/YourDisk/paddlex_cache ~/.paddlex
```

首次运行会自动下载约 200MB 模型文件。

### 硬件模式

| 模式 | 说明 | 适用 |
|------|------|------|
| `auto` | 自动检测（推荐） | 所有平台 |
| `gpu_metal` | Apple Silicon Metal 加速 | Mac M1/M2/M3 |
| `gpu_cuda` | NVIDIA CUDA 加速 | Windows/Linux + GPU |
| `cpu` | 纯 CPU | 所有平台 |

## 使用场景优化

### 手写会议纪要
```yaml
ocr:
  det_db_thresh: 0.25
output:
  confidence_threshold: 0.4
processing:
  denoise: true
  enhance_contrast: true
```

### 清晰扫描件
```yaml
processing:
  pdf_dpi: 250
  image_preprocessing: false
output:
  confidence_threshold: 0.6
```

### 模糊文档
```yaml
processing:
  pdf_dpi: 350
  denoise: true
  enhance_contrast: true
```

## Python API

```python
from ocr_extractor import DocumentExtractor

# 初始化
extractor = DocumentExtractor('config.yaml')

# 处理文件
result = extractor.extract('document.pdf')

# 访问结果
for item in result['results']:
    print(f"{item['text']} (置信度: {item['confidence']:.2f})")

# 批量处理
results = extractor.batch_extract(['file1.pdf', 'file2.jpg'])
```

## 性能参考

处理 100 页 PDF (300 DPI):

| 硬件 | 时间 |
|------|------|
| Mac M3 Max / RTX 4090 | ~1-1.5 分钟 |
| Mac M2 Pro / RTX 3080 | ~2 分钟 |
| Mac M1 / RTX 3060 | ~3-4 分钟 |
| Intel i7 CPU | ~10-15 分钟 |

## 常见问题

**Q: 如何提升识别准确率？**
- 降低置信度阈值 (0.3-0.4)
- 启用图像预处理
- 提高 PDF DPI (350-400)

**Q: 如何提升处理速度？**
- 使用 GPU 加速
- 降低 PDF DPI (200-250)
- 增加 CPU 线程数

**Q: 界面无法启动？**
```bash
# 检查依赖
pip install gradio>=4.0.0

# 更换端口
python ocr_extractor.py --web --port 8080
```

**Q: 支持哪些语言？**
- 中英文混合 (ch)
- 英文 (en)
- 法语 (french)
- 德语 (german)
- 韩语 (korean)
- 日语 (japan)

## 输出格式

**JSON:**
```json
{
  "file": "document.pdf",
  "processing_time": 12.5,
  "total_items": 45,
  "results": [
    {
      "text": "识别的文本",
      "confidence": 0.95,
      "page": 1
    }
  ]
}
```

**TXT:**
```
文件: document.pdf
处理时间: 12.50 秒
提取项数: 45

识别的文本
置信度: 0.95
```

## 项目结构

```
doc-arena/
├── ocr_extractor.py       # 主程序（包含 Web 界面和命令行）
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖清单
└── README.md             # 本文档
```

## 许可证

MIT License. PaddleOCR 使用 Apache 2.0 License.

---

**立即开始:**
```bash
python ocr_extractor.py
```
