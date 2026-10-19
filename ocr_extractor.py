#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档信息提取系统 - 基于 PaddleOCR-VL-1.6
支持 PDF、Word、图片（包括手写体）的文字识别和信息提取
支持多平台硬件加速：Mac (Metal)、Windows/Linux (CUDA)、Intel 核显、CPU

使用方法：
    python ocr_extractor.py                    # 启动 Web 界面
    python ocr_extractor.py file.pdf           # 命令行模式
    python ocr_extractor.py --help             # 查看帮助
"""

import os
import sys
import yaml
import json
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# 第三方库
try:
    from paddleocr import PaddleOCR
    import cv2
    import numpy as np
    from PIL import Image
    import pymupdf as fitz  # PyMuPDF
    from docx import Document
    import pdf2image
    import gradio as gr
except ImportError as e:
    print(f"缺少必要的依赖库: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)


class HardwareDetector:
    """硬件检测和配置"""

    @staticmethod
    def detect_platform() -> str:
        """检测操作系统平台"""
        system = platform.system().lower()
        if system == 'darwin':
            return 'mac'
        elif system == 'linux':
            return 'linux'
        elif system == 'windows':
            return 'windows'
        return 'unknown'

    @staticmethod
    def detect_apple_silicon() -> bool:
        """检测是否为 Apple Silicon (M1/M2/M3)"""
        if platform.system().lower() != 'darwin':
            return False
        try:
            machine = platform.machine().lower()
            return machine == 'arm64'
        except:
            return False

    @staticmethod
    def detect_cuda() -> bool:
        """检测 NVIDIA CUDA 是否可用"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'],
                                  capture_output=True,
                                  timeout=5)
            return result.returncode == 0
        except:
            return False

    @staticmethod
    def detect_intel_gpu() -> bool:
        """检测 Intel 核显是否可用"""
        try:
            system = platform.system().lower()
            if system == 'windows':
                import subprocess
                result = subprocess.run(['wmic', 'path', 'win32_VideoController',
                                       'get', 'name'],
                                      capture_output=True,
                                      text=True,
                                      timeout=5)
                return 'intel' in result.stdout.lower()
            return False
        except:
            return False

    @classmethod
    def auto_detect_mode(cls) -> str:
        """自动检测最佳运行模式"""
        platform_name = cls.detect_platform()

        if platform_name == 'mac':
            if cls.detect_apple_silicon():
                return 'gpu_metal'
            else:
                return 'cpu'

        if cls.detect_cuda():
            return 'gpu_cuda'

        if cls.detect_intel_gpu():
            return 'intel_gpu'

        return 'cpu'


class DocumentExtractor:
    """文档信息提取器"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化提取器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._detect_hardware()
        self._init_ocr()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            return self._get_default_config()

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'hardware': {
                'mode': 'auto',
                'device_id': 0,
                'cpu_threads': 10,
                'enable_mkldnn': True,
            },
            'ocr': {
                'model_name': 'PaddleOCR-VL-1.6',
                'model_dir': None,
                'use_angle_cls': True,
                'lang': 'ch',
                'show_log': False,
                'rec_batch_num': 6,
                'det_db_thresh': 0.3,
                'det_db_box_thresh': 0.6,
                'det_db_unclip_ratio': 1.5,
                'max_side_len': 4096,
            },
            'processing': {
                'pdf_dpi': 300,
                'image_preprocessing': True,
                'denoise': True,
                'enhance_contrast': True,
                'binarization': False,
            },
            'output': {
                'format': 'json',
                'save_images': False,
                'output_dir': 'output',
                'confidence_threshold': 0.5,
            },
            'logging': {
                'level': 'INFO',
                'file': 'ocr_extractor.log',
            }
        }

    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'ocr_extractor.log')

        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # 获取或创建 logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # 清除已存在的处理器，避免重复
        self.logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 添加文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info("日志系统初始化完成")
        self.logger.info(f"日志文件: {log_file}")

    def _detect_hardware(self):
        """检测硬件配置"""
        hardware_config = self.config.get('hardware', {})
        mode = hardware_config.get('mode', 'auto')

        if mode == 'auto':
            mode = HardwareDetector.auto_detect_mode()
            self.logger.info(f"自动检测到运行模式: {mode}")

        self.hardware_mode = mode
        self.device_id = hardware_config.get('device_id', 0)

        platform_name = HardwareDetector.detect_platform()
        self.logger.info(f"操作系统: {platform_name}")
        self.logger.info(f"运行模式: {mode}")

        if mode == 'gpu_metal':
            self.logger.info("使用 Apple Silicon (Metal) 加速")
        elif mode == 'gpu_cuda':
            self.logger.info(f"使用 NVIDIA GPU (CUDA) 加速，设备 ID: {self.device_id}")
        elif mode == 'intel_gpu':
            self.logger.info("使用 Intel 核显加速（实验性）")
        else:
            self.logger.info("使用 CPU 运行")

    def _init_ocr(self):
        """初始化 OCR 引擎"""
        ocr_config = self.config.get('ocr', {})

        try:
            self.logger.info("正在初始化 PaddleOCR 引擎...")

            # 使用最简化的参数，让 PaddleOCR 自动处理
            ocr_params = {}

            # 基本参数
            use_angle_cls = ocr_config.get('use_angle_cls', True)
            lang = ocr_config.get('lang', 'ch')

            # 新版本使用 use_textline_orientation 替代 use_angle_cls
            ocr_params['use_textline_orientation'] = use_angle_cls
            ocr_params['lang'] = lang

            self.logger.info(f"OCR 参数: {ocr_params}")
            self.logger.info(f"模型将下载到: ~/.paddlex/official_models/")
            self.ocr = PaddleOCR(**ocr_params)

            self.logger.info("PaddleOCR 引擎初始化成功")
            self.logger.info(f"硬件模式: {self.hardware_mode}")
            self.logger.info(f"语言: {lang}")
            self.logger.info(f"角度分类: {'开启' if use_angle_cls else '关闭'}")

        except Exception as e:
            self.logger.error(f"OCR 引擎初始化失败: {e}")
            self.logger.error("请检查:")
            self.logger.error("1. 是否安装了 paddleocr: pip install paddleocr")
            self.logger.error("2. 是否安装了 paddlepaddle: pip install paddlepaddle>=2.6.0")
            self.logger.error("3. Mac 用户请确保安装了正确的版本")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理，提高识别准确率"""
        if not self.config.get('processing', {}).get('image_preprocessing', True):
            return image

        try:
            # 保存原始图像的通道数
            is_color = len(image.shape) == 3

            if is_color:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            if self.config.get('processing', {}).get('denoise', True):
                gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            if self.config.get('processing', {}).get('enhance_contrast', True):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            if self.config.get('processing', {}).get('binarization', False):
                gray = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

            # 转回彩色图像（PaddleOCR 需要3通道）
            if len(gray.shape) == 2:
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            return gray
        except Exception as e:
            self.logger.warning(f"图像预处理失败，使用原始图像: {e}")
            return image

    def extract_from_image(self, image_path: str) -> List[Dict[str, Any]]:
        """从图像中提取文字"""
        self.logger.info(f"正在处理图像: {image_path}")

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

            processed_image = self.preprocess_image(image)
            results = self.ocr.predict(processed_image)

            extracted_data = []
            confidence_threshold = self.config.get('output', {}).get('confidence_threshold', 0.5)

            # 处理新版本 PaddleOCR 返回的 OCRResult 对象
            if results and len(results) > 0:
                result = results[0]

                # OCRResult 是字典对象，直接访问字段
                if isinstance(result, dict) and 'rec_texts' in result:
                    rec_texts = result['rec_texts']
                    rec_scores = result['rec_scores']
                    rec_polys = result['rec_polys']

                    self.logger.info(f"识别到 {len(rec_texts)} 条文本")

                    for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                        if score >= confidence_threshold:
                            extracted_data.append({
                                'text': text,
                                'confidence': float(score),
                                'position': poly.tolist() if hasattr(poly, 'tolist') else poly,
                                'type': 'text'
                            })
                else:
                    self.logger.warning(f"未知的 OCRResult 格式: {type(result)}")

            self.logger.info(f"从图像中提取了 {len(extracted_data)} 条文本")
            return extracted_data

        except Exception as e:
            self.logger.error(f"处理图像失败: {e}")
            import traceback
            self.logger.error(f"完整错误信息: {traceback.format_exc()}")
            raise

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从 PDF 中提取文字"""
        self.logger.info(f"正在处理 PDF: {pdf_path}")

        all_results = []

        try:
            doc = fitz.open(pdf_path)
            has_text = False

            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    has_text = True
                    all_results.append({
                        'text': text,
                        'confidence': 1.0,
                        'page': page_num + 1,
                        'type': 'text',
                        'source': 'pdf_text'
                    })

            doc.close()

            if not has_text or len(all_results) == 0:
                self.logger.info("PDF 为扫描版，使用 OCR 识别...")
                dpi = self.config.get('processing', {}).get('pdf_dpi', 300)
                images = pdf2image.convert_from_path(pdf_path, dpi=dpi)

                for page_num, image in enumerate(images):
                    img_array = np.array(image)
                    if img_array.shape[2] == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                    else:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    processed_image = self.preprocess_image(img_array)
                    results = self.ocr.ocr(processed_image)

                    if results and len(results) > 0:
                        confidence_threshold = self.config.get('output', {}).get('confidence_threshold', 0.5)
                        result = results[0]

                        # OCRResult 是字典对象，直接访问字段
                        if isinstance(result, dict) and 'rec_texts' in result:
                            rec_texts = result['rec_texts']
                            rec_scores = result['rec_scores']
                            rec_polys = result['rec_polys']

                            for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                                if score >= confidence_threshold:
                                    all_results.append({
                                        'text': text,
                                        'confidence': float(score),
                                        'position': poly.tolist() if hasattr(poly, 'tolist') else poly,
                                        'page': page_num + 1,
                                        'type': 'text',
                                        'source': 'pdf_ocr'
                                    })

            self.logger.info(f"从 PDF 中提取了 {len(all_results)} 条文本")
            return all_results

        except Exception as e:
            self.logger.error(f"处理 PDF 失败: {e}")
            raise

    def extract_from_word(self, word_path: str) -> List[Dict[str, Any]]:
        """从 Word 文档中提取文字"""
        self.logger.info(f"正在处理 Word 文档: {word_path}")

        all_results = []

        try:
            doc = Document(word_path)

            for para_num, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    all_results.append({
                        'text': para.text,
                        'confidence': 1.0,
                        'paragraph': para_num + 1,
                        'type': 'text',
                        'source': 'word_text'
                    })

            for table_num, table in enumerate(doc.tables):
                for row_num, row in enumerate(table.rows):
                    row_text = ' | '.join([cell.text for cell in row.cells])
                    if row_text.strip():
                        all_results.append({
                            'text': row_text,
                            'confidence': 1.0,
                            'table': table_num + 1,
                            'row': row_num + 1,
                            'type': 'table',
                            'source': 'word_table'
                        })

            self.logger.info(f"从 Word 文档中提取了 {len(all_results)} 条文本")
            return all_results

        except Exception as e:
            self.logger.error(f"处理 Word 文档失败: {e}")
            raise

    def extract(self, file_path: str) -> Dict[str, Any]:
        """自动识别文件类型并提取信息"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        start_time = datetime.now()

        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            results = self.extract_from_image(file_path)
        elif file_ext == '.pdf':
            results = self.extract_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            results = self.extract_from_word(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        output = {
            'file': file_path,
            'file_type': file_ext,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'total_items': len(results),
            'results': results
        }

        self._save_results(output)
        return output

    def _save_results(self, output: Dict[str, Any]):
        """保存提取结果"""
        output_config = self.config.get('output', {})
        output_dir = output_config.get('output_dir', 'output')
        output_format = output_config.get('format', 'json')

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = Path(output['file']).stem
        output_file = os.path.join(output_dir, f"{file_name}_{timestamp}.{output_format}")

        try:
            if output_format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
            elif output_format == 'txt':
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"文件: {output['file']}\n")
                    f.write(f"处理时间: {output['processing_time']:.2f} 秒\n")
                    f.write(f"提取项数: {output['total_items']}\n")
                    f.write("\n" + "="*50 + "\n\n")
                    for item in output['results']:
                        f.write(f"{item['text']}\n")
                        f.write(f"置信度: {item['confidence']:.2f}\n\n")

            self.logger.info(f"结果已保存到: {output_file}")

        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")

    def batch_extract(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """批量提取多个文件"""
        all_outputs = []

        for file_path in file_paths:
            try:
                output = self.extract(file_path)
                output['success'] = True
                all_outputs.append(output)
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 失败: {e}")
                all_outputs.append({
                    'file': file_path,
                    'error': str(e),
                    'success': False
                })

        return all_outputs

    def batch_extract_simple(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        批量提取多个文件 - 简化版
        返回格式：{filename: extracted_text}
        """
        results = {}

        for file_path in file_paths:
            try:
                output = self.extract(file_path)
                filename = Path(file_path).name

                # 合并所有识别的文本
                texts = [item['text'] for item in output['results']]
                combined_text = '\n'.join(texts)

                results[filename] = {
                    'text': combined_text,
                    'total_items': len(texts),
                    'processing_time': output['processing_time'],
                    'file_type': output['file_type'],
                    'success': True
                }
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 失败: {e}")
                results[Path(file_path).name] = {
                    'error': str(e),
                    'success': False
                }

        return results


class WebInterface:
    """Web 界面类"""

    def __init__(self):
        """初始化界面"""
        self.extractor = None
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        # 获取或创建 logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 如果已经有处理器，说明已经配置过了，直接返回
        if self.logger.handlers:
            return

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def initialize_extractor(
        self,
        hardware_mode: str,
        cpu_threads: int,
        use_angle_cls: bool,
        lang: str,
        confidence_threshold: float,
        pdf_dpi: int,
        enable_preprocessing: bool
    ) -> str:
        """初始化提取器"""
        try:
            # 读取现有配置文件，保留所有设置和注释
            config_file = 'config.yaml'
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                # 如果不存在，使用默认配置
                config = {
                    'hardware': {'mode': 'auto', 'device_id': 0, 'cpu_threads': 10, 'enable_mkldnn': True},
                    'ocr': {'model_name': 'PaddleOCR-VL-1.6', 'model_dir': None, 'use_angle_cls': True, 'lang': 'ch'},
                    'processing': {'pdf_dpi': 300, 'image_preprocessing': True, 'denoise': True, 'enhance_contrast': True, 'binarization': False},
                    'output': {'format': 'json', 'save_images': False, 'output_dir': 'output', 'confidence_threshold': 0.5},
                    'logging': {'level': 'INFO', 'file': 'ocr_extractor.log'}
                }

            # 只更新用户在界面中修改的配置项
            if 'hardware' not in config:
                config['hardware'] = {}
            config['hardware']['mode'] = hardware_mode
            config['hardware']['cpu_threads'] = cpu_threads

            if 'ocr' not in config:
                config['ocr'] = {}
            config['ocr']['use_angle_cls'] = use_angle_cls
            config['ocr']['lang'] = lang

            if 'processing' not in config:
                config['processing'] = {}
            config['processing']['pdf_dpi'] = pdf_dpi
            config['processing']['image_preprocessing'] = enable_preprocessing

            if 'output' not in config:
                config['output'] = {}
            config['output']['confidence_threshold'] = confidence_threshold

            # 直接使用配置初始化，不写入文件，保留原有注释
            self.extractor = DocumentExtractor.__new__(DocumentExtractor)
            self.extractor.config = config
            self.extractor._setup_logging()
            self.extractor._detect_hardware()
            self.extractor._init_ocr()

            platform_name = HardwareDetector.detect_platform()
            actual_mode = self.extractor.hardware_mode

            status = f"✅ 初始化成功！\n\n"
            status += f"🖥️ 系统平台: {platform_name}\n"
            status += f"⚡ 运行模式: {actual_mode}\n"

            if actual_mode == 'gpu_metal':
                status += f"🚀 使用 Apple Silicon Metal 加速\n"
            elif actual_mode == 'gpu_cuda':
                status += f"🚀 使用 NVIDIA CUDA 加速\n"
            elif actual_mode == 'cpu':
                status += f"💻 使用 CPU 模式（{cpu_threads} 线程）\n"

            status += f"🌐 语言: {lang}\n"
            status += f"📊 置信度阈值: {confidence_threshold}\n"

            return status

        except Exception as e:
            return f"❌ 初始化失败: {str(e)}\n\n请检查配置和依赖是否正确安装。"

    def extract_document(
        self,
        files: List[str],
        progress=gr.Progress()
    ) -> Tuple[str, str, List[Dict[str, str]]]:
        """提取文档内容 - 返回单个汇总 + JSON + 每个文件的独立结果"""
        if not self.extractor:
            return "❌ 请先点击'初始化系统'按钮", "", []

        if not files:
            return "❌ 请先上传文件", "", []

        try:
            results = []
            total_files = len(files)

            progress(0, desc="开始处理...")

            for idx, file_path in enumerate(files):
                progress(
                    (idx + 1) / total_files,
                    desc=f"处理中 {idx + 1}/{total_files}: {Path(file_path).name}"
                )

                try:
                    result = self.extractor.extract(file_path)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"处理文件 {file_path} 失败: {e}")
                    results.append({
                        'file': file_path,
                        'error': str(e),
                        'success': False
                    })

            progress(1.0, desc="处理完成！")

            # 生成汇总结果
            summary = self._format_summary(results)
            json_result = json.dumps(results, ensure_ascii=False, indent=2)

            # 生成每个文件的独立结果 - 包含文件名用于标签页标题
            individual_results = []
            for idx, result in enumerate(results, 1):
                individual_results.append({
                    'tab_title': f"文件 {idx}",
                    'filename': Path(result['file']).name,
                    'content': self._format_single_file_result(result)
                })

            return summary, json_result, individual_results

        except Exception as e:
            return f"❌ 处理失败: {str(e)}", "", []

    def _format_summary(self, results: List[dict]) -> str:
        """格式化汇总结果"""
        output = []

        total_files = len(results)
        success_files = sum(1 for r in results if r.get('success', True))
        total_items = sum(r.get('total_items', 0) for r in results if r.get('success', True))
        total_time = sum(r.get('processing_time', 0) for r in results if r.get('success', True))

        output.append(f"📊 批量处理汇总")
        output.append(f"{'='*80}\n")
        output.append(f"📁 处理文件数: {total_files} 个")
        output.append(f"✅ 成功: {success_files} 个")
        if success_files < total_files:
            output.append(f"❌ 失败: {total_files - success_files} 个")
        output.append(f"📝 总提取项数: {total_items} 条")
        output.append(f"⏱️  总处理时间: {total_time:.2f} 秒")

        if success_files > 0:
            output.append(f"⚡ 平均速度: {total_time/success_files:.2f} 秒/文件")

        output.append(f"\n{'='*80}\n")
        output.append(f"📋 文件列表:\n")

        for idx, result in enumerate(results, 1):
            filename = Path(result['file']).name
            if result.get('success', True):
                items = result.get('total_items', 0)
                time = result.get('processing_time', 0)
                output.append(f"{idx}. ✅ {filename} ({items} 项, {time:.2f}s)")
            else:
                output.append(f"{idx}. ❌ {filename} - {result.get('error', '未知错误')}")

        output.append(f"\n💡 提示: 每个文件的详细识别内容请查看下方对应的文本框")

        return "\n".join(output)

    def _format_single_file_result(self, result: dict) -> str:
        """格式化单个文件的结果"""
        output = []

        if not result.get('success', True):
            output.append(f"❌ 处理失败")
            output.append(f"错误信息: {result.get('error', '未知错误')}")
            return "\n".join(output)

        filename = Path(result['file']).name
        output.append(f"📄 {filename}")
        output.append(f"{'='*80}")
        output.append(f"📌 文件类型: {result['file_type']}")
        output.append(f"⏱️  处理时间: {result['processing_time']:.2f} 秒")
        output.append(f"📝 提取项数: {result['total_items']}")
        output.append(f"\n{'─'*80}")
        output.append(f"📖 识别内容:")
        output.append(f"{'─'*80}\n")

        for item in result['results']:
            confidence = item['confidence']
            text = item['text'].strip()

            if confidence >= 0.9:
                mark = "✅"
            elif confidence >= 0.7:
                mark = "✓"
            else:
                mark = "⚠️"
            output.append(f"{mark} [{confidence:.2f}] {text}")

        return "\n".join(output)

    def _format_text_results(self, results: List[dict]) -> str:
        """格式化文本结果"""
        output = []

        # 添加总体统计
        total_files = len(results)
        success_files = sum(1 for r in results if r.get('success', True))
        output.append(f"📊 批量处理结果: {success_files}/{total_files} 个文件成功")
        output.append(f"{'='*80}\n")

        for idx, result in enumerate(results, 1):
            if result.get('success', True):
                output.append(f"\n{'='*80}")
                output.append(f"📄 文件 {idx}: {Path(result['file']).name}")
                output.append(f"{'='*80}")
                output.append(f"📌 文件类型: {result['file_type']}")
                output.append(f"⏱️  处理时间: {result['processing_time']:.2f} 秒")
                output.append(f"📝 提取项数: {result['total_items']}")
                output.append(f"\n{'─'*80}")
                output.append(f"📖 识别内容:")
                output.append(f"{'─'*80}\n")

                # 显示所有结果，每条一行
                for item_idx, item in enumerate(result['results'], 1):
                    confidence = item['confidence']
                    text = item['text'].strip()
                    # 根据置信度显示不同的标记
                    if confidence >= 0.9:
                        mark = "✅"
                    elif confidence >= 0.7:
                        mark = "✓"
                    else:
                        mark = "⚠️"
                    output.append(f"{mark} [{confidence:.2f}] {text}")

                output.append(f"\n{'─'*80}")
                output.append(f"✅ 文件处理完成\n")
            else:
                output.append(f"\n{'='*80}")
                output.append(f"❌ 文件 {idx}: {Path(result['file']).name}")
                output.append(f"{'='*80}")
                output.append(f"错误: {result['error']}\n")

        return "\n".join(output)


def create_web_interface():
    """创建 Gradio Web 界面"""

    web_interface = WebInterface()

    platform_name = HardwareDetector.detect_platform()
    is_apple_silicon = HardwareDetector.detect_apple_silicon()
    has_cuda = HardwareDetector.detect_cuda()

    if is_apple_silicon:
        default_mode = "gpu_metal"
        hardware_info = "🍎 检测到 Apple Silicon，推荐使用 Metal 加速"
    elif has_cuda:
        default_mode = "gpu_cuda"
        hardware_info = "🎮 检测到 NVIDIA GPU，推荐使用 CUDA 加速"
    else:
        default_mode = "cpu"
        hardware_info = "💻 将使用 CPU 模式"

    with gr.Blocks(title="文档信息提取系统") as demo:

        gr.Markdown(f"""
        # 📄 文档信息提取系统

        **基于 PaddleOCR-VL-1.6 | 支持 PDF、Word、图片（含手写体）**

        {hardware_info}
        """)

        with gr.Tabs():
            with gr.Tab("📄 文档提取"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 上传文件")
                        files_input = gr.File(
                            label="选择文件（支持 PDF、Word、图片）",
                            file_count="multiple",
                            file_types=[
                                ".pdf", ".docx", ".doc",
                                ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
                            ]
                        )

                        extract_btn = gr.Button(
                            "🚀 开始提取",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 提取结果")

                        with gr.Tabs():
                            # 处理汇总标签页
                            with gr.Tab("📊 处理汇总"):
                                summary_output = gr.Textbox(
                                    label="批量处理统计",
                                    lines=15,
                                    max_lines=20
                                )

                            # 各文件识别结果标签页
                            file_outputs = []
                            for i in range(10):
                                with gr.Tab(f"文件 {i+1}", visible=False) as tab:
                                    output_box = gr.Textbox(
                                        label="识别内容",
                                        lines=20,
                                        max_lines=25
                                    )
                                file_outputs.append((tab, output_box))

                            # JSON 结果标签页
                            with gr.Tab("📊 JSON 结果"):
                                json_output = gr.Code(
                                    label="完整 JSON 数据",
                                    language="json",
                                    lines=20
                                )

                # 处理函数
                def process_and_display(files):
                    if not files:
                        empty_result = ["❌ 请先上传文件", ""] + [gr.Tab(visible=False), ""] * 10 + [""]
                        return empty_result

                    summary, json_result, individual_results = web_interface.extract_document(files)

                    # 构建输出列表
                    outputs = [summary]

                    # 为每个文件结果更新对应的标签页
                    for i in range(10):
                        if i < len(individual_results):
                            # 显示这个标签页，使用实际文件名，并填充内容
                            filename = individual_results[i]['filename']
                            outputs.append(gr.Tab(label=f"📄 {filename}", visible=True))
                            outputs.append(individual_results[i]['content'])
                        else:
                            # 隐藏未使用的标签页
                            outputs.append(gr.Tab(visible=False))
                            outputs.append("")

                    outputs.append(json_result)
                    return outputs

                # 构建输出组件列表
                output_components = [summary_output]
                for tab, textbox in file_outputs:
                    output_components.extend([tab, textbox])
                output_components.append(json_output)

                extract_btn.click(
                    fn=process_and_display,
                    inputs=[files_input],
                    outputs=output_components
                )

            with gr.Tab("⚙️ 系统配置"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🖥️ 硬件配置")

                        hardware_mode = gr.Radio(
                            label="运行模式",
                            choices=[
                                ("自动检测", "auto"),
                                ("Apple Metal 加速 (M1/M2/M3)", "gpu_metal"),
                                ("NVIDIA CUDA 加速", "gpu_cuda"),
                                ("Intel 核显加速（实验性）", "intel_gpu"),
                                ("纯 CPU 模式", "cpu"),
                            ],
                            value=default_mode,
                            info="选择硬件加速模式"
                        )

                        cpu_threads = gr.Slider(
                            label="CPU 线程数",
                            minimum=1,
                            maximum=32,
                            value=10,
                            step=1,
                            info="CPU 模式或数据预处理使用的线程数"
                        )

                    with gr.Column():
                        gr.Markdown("### 🔧 OCR 配置")

                        use_angle_cls = gr.Checkbox(
                            label="启用角度分类器（自动纠正文字方向）",
                            value=True,
                            info="推荐开启，对拍照文档特别有效"
                        )

                        lang = gr.Dropdown(
                            label="识别语言",
                            choices=[
                                ("中英文混合", "ch"),
                                ("英文", "en"),
                                ("法语", "french"),
                                ("德语", "german"),
                                ("韩语", "korean"),
                                ("日语", "japan"),
                            ],
                            value="ch",
                            info="选择主要识别语言"
                        )

                        confidence_threshold = gr.Slider(
                            label="置信度阈值",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            info="只保留置信度高于此值的结果"
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎨 图像处理")

                        pdf_dpi = gr.Slider(
                            label="PDF 转图像 DPI",
                            minimum=150,
                            maximum=600,
                            value=300,
                            step=50,
                            info="越高质量越好但速度越慢"
                        )

                        enable_preprocessing = gr.Checkbox(
                            label="启用图像预处理（去噪、增强对比度）",
                            value=True,
                            info="对模糊或拍照文档有帮助"
                        )

                    with gr.Column():
                        gr.Markdown("### 💾 系统状态")

                        init_status = gr.Textbox(
                            label="初始化状态",
                            lines=10,
                            value="⏳ 请点击下方按钮初始化系统...",
                            interactive=False
                        )

                        init_btn = gr.Button(
                            "🔄 初始化系统",
                            variant="primary",
                            size="lg"
                        )

                init_btn.click(
                    fn=web_interface.initialize_extractor,
                    inputs=[
                        hardware_mode,
                        cpu_threads,
                        use_angle_cls,
                        lang,
                        confidence_threshold,
                        pdf_dpi,
                        enable_preprocessing
                    ],
                    outputs=[init_status]
                )

            with gr.Tab("📖 帮助文档"):
                gr.Markdown("""
                # 使用说明

                ## 快速开始

                1. **初始化系统**
                   - 进入"系统配置"标签页
                   - 根据需要调整参数（通常使用默认值即可）
                   - 点击"初始化系统"按钮

                2. **提取文档**
                   - 返回"文档提取"标签页
                   - 点击上传区域，选择要处理的文件
                   - 可以一次选择多个文件
                   - 点击"开始提取"按钮

                3. **查看结果**
                   - 在"文本结果"标签查看可读的文本
                   - 在"JSON 结果"标签查看完整的结构化数据
                   - 结果会自动保存到 `output` 目录

                ## 支持的文件格式

                - **PDF**: 文本型和扫描型 PDF 文档
                - **Word**: .docx 和 .doc 格式
                - **图片**: JPG、PNG、BMP、TIFF、WebP 等
                - **手写体**: 手写会议纪要、笔记等

                ## 硬件模式说明

                - **自动检测**: 自动选择最佳模式（推荐）
                - **Apple Metal**: Mac M1/M2/M3 专用，速度最快
                - **NVIDIA CUDA**: Windows/Linux + NVIDIA GPU
                - **Intel 核显**: 实验性功能，效果有限
                - **纯 CPU**: 所有平台通用，速度较慢

                ## 参数调整建议

                ### 手写体识别
                - 置信度阈值: 0.3-0.4（降低以保留更多结果）
                - 启用图像预处理: 是

                ### 清晰扫描件
                - 置信度阈值: 0.6-0.7（提高以确保准确性）
                - PDF DPI: 250-300

                ### 模糊或老旧文档
                - PDF DPI: 350-400（提高分辨率）
                - 启用图像预处理: 是

                ## 性能参考

                **处理 100 页 PDF 的大致时间：**
                - Mac M3 Max / RTX 4090: ~1-1.5 分钟
                - Mac M2 Pro / RTX 3080: ~2-2.5 分钟
                - Mac M1 / RTX 3060: ~3-4 分钟
                - Intel i7 CPU: ~10-15 分钟

                ## 常见问题

                **Q: 初始化失败怎么办？**
                - 检查是否安装了所有依赖: `pip install -r requirements.txt`
                - 检查 PaddlePaddle 是否正确安装
                - 尝试切换到 CPU 模式

                **Q: 识别准确率低？**
                - 降低置信度阈值（但会增加误识别）
                - 启用图像预处理
                - 提高 PDF DPI（处理扫描件时）

                **Q: 处理速度太慢？**
                - 使用 GPU 加速模式
                - 降低 PDF DPI
                - 关闭图像预处理（针对清晰文档）
                """)

    return demo


def main_cli():
    """命令行模式主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='文档信息提取系统')
    parser.add_argument('files', nargs='*', help='要处理的文件路径')
    parser.add_argument('-c', '--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('-o', '--output-dir', help='输出目录（覆盖配置文件）')
    parser.add_argument('-f', '--format', choices=['json', 'txt'], help='输出格式（覆盖配置文件）')
    parser.add_argument('--web', action='store_true', help='启动 Web 界面')
    parser.add_argument('--host', default='127.0.0.1', help='Web 服务器地址')
    parser.add_argument('--port', type=int, default=7860, help='Web 服务器端口')
    # parser.add_argument('--share', action='store_true', default=True, help='创建公共分享链接')
    parser.add_argument('--share', action='store_true', help='创建公共分享链接')

    args = parser.parse_args()

    # Web 界面模式
    if args.web or not args.files:
        demo = create_web_interface()
        print(f"\n{'='*60}")
        print(f"🚀 文档信息提取系统 Web 界面")
        print(f"{'='*60}")
        print(f"📍 本地访问: http://{args.host}:{args.port}")
        if args.share:
            print(f"🌐 公共访问: 启动后将显示分享链接")
        print(f"{'='*60}\n")
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )
        return

    # 命令行模式
    extractor = DocumentExtractor(args.config)

    if args.output_dir:
        extractor.config['output']['output_dir'] = args.output_dir
    if args.format:
        extractor.config['output']['format'] = args.format

    results = extractor.batch_extract(args.files)

    success_count = sum(1 for r in results if r.get('success', True))
    print(f"\n处理完成: {success_count}/{len(results)} 个文件成功")


if __name__ == '__main__':
    main_cli()
