#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
咸鱼之王实时关卡识别工具
用于识别游戏窗口中的关卡数字，输出第xxx关
"""

import os
import sys
import cv2
import numpy as np
import re
import time
from datetime import datetime
from PIL import ImageGrab
import win32gui
import win32con
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置
from config import Config

# 创建调试目录
debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_screenshots')
os.makedirs(debug_dir, exist_ok=True)

print("====== 咸鱼之王实时关卡识别工具 ======")
print(f"游戏窗口: {Config.GAME_WINDOW_TITLE}")
print(f"调试目录: {debug_dir}")
print("按Ctrl+C停止识别")
print("======================================\n")

# 尝试导入OCR
oocr_available = False
reader = None

# 仅在直接运行时初始化OCR引擎
if __name__ == "__main__":
    try:
        import easyocr
        print("正在初始化OCR引擎...")
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=Config.OCR_GPU_ENABLED)
        ocr_available = True
        print("OCR引擎初始化成功")
    except Exception as e:
        print(f"OCR引擎初始化失败: {e}")
        print("请安装easyocr: pip install easyocr")
        sys.exit(1)


class LevelRecognizer:
    """关卡识别器
    
    支持作为模块导入使用，可以传入外部OCR引擎实例
    """
    
    def __init__(self, external_reader=None):
        self.hwnd = None
        self.last_level = None
        self.recognition_count = 0
        self.success_count = 0
        # 使用外部提供的OCR引擎或使用全局的
        self.reader = external_reader if external_reader else reader
        # 如果没有可用的OCR引擎，尝试初始化
        if not self.reader:
            try:
                import easyocr
                print("[LevelRecognizer] 初始化OCR引擎...")
                self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=Config.OCR_GPU_ENABLED)
                print("[LevelRecognizer] OCR引擎初始化成功")
            except Exception as e:
                print(f"[LevelRecognizer] OCR引擎初始化失败: {e}")
                self.reader = None
    
    def find_game_window(self):
        """查找游戏窗口"""
        print(f"正在查找游戏窗口: {Config.GAME_WINDOW_TITLE}")
        self.hwnd = win32gui.FindWindow(None, Config.GAME_WINDOW_TITLE)
        if self.hwnd:
            print(f"✓ 找到游戏窗口，句柄: {self.hwnd}")
            # 确保窗口可见，但不尝试设置为前台
            try:
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            except Exception as e:
                print(f"提示: 窗口可见性设置可能受限: {e}")
            return True
        else:
            print(f"✗ 未找到游戏窗口: {Config.GAME_WINDOW_TITLE}")
            return False
    
    def get_level_region(self):
        """获取关卡识别区域（屏幕坐标）"""
        try:
            region = Config.LEVEL_RECOGNITION_REGION
            (x1, y1) = win32gui.ClientToScreen(self.hwnd, (region['left'], region['top']))
            (x2, y2) = win32gui.ClientToScreen(self.hwnd, (region['right'], region['bottom']))
            screen_region = (x1, y1, x2, y2)
            return screen_region
        except Exception as e:
            print(f"获取识别区域出错: {e}")
            return None
    
    def capture_and_recognize(self, region, save_debug=True):
        """捕获并识别关卡
        
        Args:
            region: 屏幕区域坐标 (x1, y1, x2, y2)
            save_debug: 是否保存调试图像
            
        Returns:
            tuple: (识别到的关卡数字, 置信度)
        """
        self.recognition_count += 1
        
        try:
            # 确保有OCR引擎可用
            if not self.reader:
                print("[LevelRecognizer] 错误: 没有可用的OCR引擎")
                return "", 0.0
            
            # 捕获屏幕区域
            img = ImageGrab.grab(bbox=region)
            img_np = np.array(img)
            
            # 保存原始捕获图像
            if save_debug:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                capture_path = os.path.join(debug_dir, f"level_capture_{timestamp}.png")
                img.save(capture_path)
                print(f"已保存捕获图像: {capture_path}")
            
            # 图像预处理 - 转换为灰度图
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            
            # 中值滤波降噪
            median_blur = cv2.medianBlur(gray, 3)
            
            # 尝试多种预处理方法
            processed_images = []
            
            # 方法1: 自适应阈值处理 (高斯)
            adaptive_gaussian = cv2.adaptiveThreshold(
                median_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 13, 3
            )
            processed_images.append(('adaptive_gaussian', adaptive_gaussian))
            
            # 方法2: 自适应阈值处理 (均值)
            adaptive_mean = cv2.adaptiveThreshold(
                median_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 13, 3
            )
            processed_images.append(('adaptive_mean', adaptive_mean))
            
            # 方法3: 固定阈值处理
            _, binary = cv2.threshold(median_blur, 127, 255, cv2.THRESH_BINARY_INV)
            processed_images.append(('binary', binary))
            
            # 方法4: Otsu阈值处理
            _, otsu = cv2.threshold(median_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            processed_images.append(('otsu', otsu))
            
            # 保存所有处理后的图像
            if save_debug:
                for name, img_proc in processed_images:
                    proc_path = os.path.join(debug_dir, f"level_{name}_{timestamp}.png")
                    cv2.imwrite(proc_path, img_proc)
            
            # 收集所有OCR结果
            all_results = []
            
            # 处理原始图像
            results_original = self.reader.readtext(img_np)
            all_results.extend([(r, 'original') for r in results_original])
            
            # 处理所有预处理图像
            for name, img_proc in processed_images:
                results = self.reader.readtext(img_proc)
                all_results.extend([(r, name) for r in results])
            
            # 调试信息：显示识别到的所有文本
            print(f"总共识别到 {len(all_results)} 个文本区域")
            
            # 提取关卡信息
            level_candidates = []
            
            # 优先查找包含"第"和"关"的模式
            for result, method in all_results:
                if len(result) >= 2:
                    text = result[1].strip()
                    confidence = result[2] if len(result) > 2 else 0
                    
                    # 只显示置信度大于0.3的结果，减少噪声
                    if confidence > 0.3:
                        print(f"[{method}] 识别到: '{text}' (置信度: {confidence:.2f})")
                    
                    # 模式1: 匹配"第xxx关"格式（最高优先级）
                    match = re.search(r'第(\d+)关', text)
                    if match:
                        digits = match.group(1)
                        # 包含"第"和"关"的结果给予最高优先级
                        score = confidence + 0.8
                        level_candidates.append((digits, score, confidence, text, method))
                    # 模式2: 如果文本中同时包含"第"和数字
                    elif '第' in text and any(char.isdigit() for char in text):
                        # 尝试提取"第"后面的数字
                        digits = re.search(r'第[^\d]*([\d]+)', text)
                        if digits:
                            digits = digits.group(1)
                            score = confidence + 0.5
                            level_candidates.append((digits, score, confidence, text, method))
                    # 模式3: 如果文本中同时包含"关"和数字
                    elif '关' in text and any(char.isdigit() for char in text):
                        # 尝试提取"关"前面的数字
                        digits = re.search(r'([\d]+)[^\d]*关', text)
                        if digits:
                            digits = digits.group(1)
                            score = confidence + 0.5
                            level_candidates.append((digits, score, confidence, text, method))
                    # 模式4: 检查是否有中文数字和阿拉伯数字的组合
                    elif any(char in text for char in '零一二三四五六七八九十百千万') and any(char.isdigit() for char in text):
                        # 提取所有阿拉伯数字
                        digits = re.sub(r'\D', '', text)
                        if digits and len(digits) >= 2:
                            score = confidence + 0.3
                            level_candidates.append((digits, score, confidence, text, method))
                    # 模式5: 纯数字（作为备选）
                    else:
                        digits = re.sub(r'\D', '', text)
                        # 关卡号通常是2-4位数字
                        if digits and 2 <= len(digits) <= 4:
                            # 优先考虑三位数的数字（更可能是关卡号）
                            score = confidence
                            if len(digits) == 3:
                                score += 0.2
                            level_candidates.append((digits, score, confidence, text, method))
            
            # 去重：如果有多个相同的数字，保留分数最高的
            unique_candidates = {}
            for digits, score, conf, text, method in level_candidates:
                if digits not in unique_candidates or score > unique_candidates[digits][1]:
                    unique_candidates[digits] = (digits, score, conf, text, method)
            
            # 按分数排序
            sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[1], reverse=True)
            
            if sorted_candidates:
                best_level, _, conf, text, method = sorted_candidates[0]
                print(f"\n最佳匹配: [{method}] '{text}' -> 提取数字: {best_level} (置信度: {conf:.2f})")
                
                # 检查是否成功识别到"第"和"关"字
                if '第' in text and '关' in text:
                    print("✓ 成功识别到'第xxx关'格式")
                elif '第' in text:
                    print("⚠ 识别到'第'字，但未识别到'关'字")
                elif '关' in text:
                    print("⚠ 识别到'关'字，但未识别到'第'字")
                else:
                    print("⚠ 未识别到'第'和'关'字，使用纯数字识别结果")
                    print("提示: 将在输出时自动添加'第'和'关'字以符合格式要求")
                    
                self.success_count += 1
                return best_level, conf
            else:
                print("\n未能找到有效的关卡信息")
                # 尝试分析为什么没有识别到，显示所有低置信度结果
                for result, method in all_results:
                    if len(result) >= 2:
                        text = result[1].strip()
                        confidence = result[2] if len(result) > 2 else 0
                        print(f"[低置信度] [{method}] '{text}' (置信度: {confidence:.2f})")
                return "", 0.0
                
        except Exception as e:
            print(f"识别过程出错: {e}")
            return "", 0.0
    
    def recognize_level_realtime(self, interval=0.5):
        """
        实时识别关卡
        
        Args:
            interval: 识别间隔（秒）
        """
        # 查找窗口
        if not self.find_game_window():
            print("请先启动游戏!")
            return
        
        # 获取识别区域
        region = self.get_level_region()
        if not region:
            print("无法获取识别区域")
            return
        
        print(f"识别区域: {region}")
        print("开始实时识别...")
        print("按Ctrl+C停止识别\n")
        
        start_time = time.time()
        last_print_time = start_time
        
        try:
            while True:
                current_time = time.time()
                
                # 识别关卡
                level, confidence = self.capture_and_recognize(region, save_debug=False)
                
                # 只有当识别到新的不同关卡时才打印
                if level and level != self.last_level:
                    self.last_level = level
                    # 确保输出格式为"第xxx关"
                    print(f"第{level}关 (置信度: {confidence:.2f})")
                
                # 每秒更新一次识别统计
                if current_time - last_print_time >= 1.0:
                    fps = self.recognition_count / (current_time - start_time + 0.001)
                    success_rate = (self.success_count / self.recognition_count * 100) if self.recognition_count > 0 else 0
                    
                    # 使用回车覆盖上一行
                    stats = f"识别统计 - FPS: {fps:.1f}, 成功率: {success_rate:.1f}% ({self.success_count}/{self.recognition_count})"
                    print(f"\r{stats}", end="", flush=True)
                    last_print_time = current_time
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n识别已停止")
        finally:
            total_time = time.time() - start_time
            success_rate = (self.success_count / self.recognition_count * 100) if self.recognition_count > 0 else 0
            
            print("\n===== 识别统计 =====")
            print(f"总识别次数: {self.recognition_count}")
            print(f"成功次数: {self.success_count}")
            print(f"成功率: {success_rate:.1f}%")
            print(f"总用时: {total_time:.2f}秒")
            print(f"平均FPS: {self.recognition_count / total_time:.1f}")
            print("==================")


def test_single_recognition():
    """单次识别测试"""
    recognizer = LevelRecognizer()
    
    if not recognizer.find_game_window():
        print("请先启动游戏!")
        return
    
    region = recognizer.get_level_region()
    if not region:
        return
    
    print(f"\n开始单次识别测试...")
    print(f"识别区域: {region}")
    
    level, confidence = recognizer.capture_and_recognize(region, save_debug=True)
    
    if level:
        # 确保输出格式为"第xxx关"
        print(f"\n✓ 识别成功: 第{level}关")
        print(f"置信度: {confidence:.2f}")
        print(f"调试图像已保存至: {debug_dir}")
    else:
        print("\n✗ 未能识别到关卡数字")
        print(f"调试图像已保存至: {debug_dir}，请检查识别区域是否正确")


def select_region_manually():
    """
    手动设置关卡识别区域
    允许用户通过输入坐标来设置识别区域
    """
    print("\n===== 手动设置识别区域 =====")
    print("提示: 关卡识别正确区域应该在当前识别区域的右下角位置")
    print("请输入游戏窗口内的相对坐标 (左上和右下)")
    
    recognizer = LevelRecognizer()
    
    # 查找游戏窗口
    if not recognizer.find_game_window():
        print("请先启动游戏!")
        return False
    
    try:
        # 获取窗口位置和大小
        rect = win32gui.GetWindowRect(recognizer.hwnd)
        x, y, width, height = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
        
        print(f"游戏窗口位置: ({x}, {y})，大小: {width}x{height}")
        print(f"当前识别区域配置:")
        print(f"- 左上角: ({Config.LEVEL_RECOGNITION_REGION['left']}, {Config.LEVEL_RECOGNITION_REGION['top']})")
        print(f"- 右下角: ({Config.LEVEL_RECOGNITION_REGION['right']}, {Config.LEVEL_RECOGNITION_REGION['bottom']})")
        
        # 提供建议坐标（当前区域的右下角位置）
        current_left = Config.LEVEL_RECOGNITION_REGION['left']
        current_top = Config.LEVEL_RECOGNITION_REGION['top']
        current_right = Config.LEVEL_RECOGNITION_REGION['right']
        current_bottom = Config.LEVEL_RECOGNITION_REGION['bottom']
        
        # 计算建议的右下角区域坐标
        # 假设关卡数字在当前区域的右下角，所以缩小区域到右下角部分
        suggestion_left = current_left + int((current_right - current_left) * 0.6)  # 从右侧60%开始
        suggestion_top = current_top + int((current_bottom - current_top) * 0.6)   # 从底部60%开始
        suggestion_right = current_right
        suggestion_bottom = current_bottom
        
        print(f"\n建议的关卡识别区域（右下角）:")
        print(f"- 左上角: ({suggestion_left}, {suggestion_top})")
        print(f"- 右下角: ({suggestion_right}, {suggestion_bottom})")
        
        # 让用户选择使用建议坐标还是手动输入
        use_suggestion = input("是否使用建议的识别区域? (y/n): ")
        
        if use_suggestion.lower() == 'y':
            rel_x1, rel_y1, rel_x2, rel_y2 = suggestion_left, suggestion_top, suggestion_right, suggestion_bottom
        else:
            print("\n请输入关卡识别区域的坐标:")
            
            # 获取用户输入的坐标
            try:
                rel_x1 = int(input(f"左上角X坐标 (当前: {current_left}): ") or current_left)
                rel_y1 = int(input(f"左上角Y坐标 (当前: {current_top}): ") or current_top)
                rel_x2 = int(input(f"右下角X坐标 (当前: {current_right}): ") or current_right)
                rel_y2 = int(input(f"右下角Y坐标 (当前: {current_bottom}): ") or current_bottom)
            except ValueError:
                print("输入无效，请输入数字坐标")
                return False
        
        # 验证坐标有效性
        if rel_x1 >= rel_x2 or rel_y1 >= rel_y2:
            print("错误: 左上角坐标必须小于右下角坐标")
            return False
        
        if rel_x1 < 0 or rel_y1 < 0 or rel_x2 > width or rel_y2 > height:
            print("警告: 坐标超出窗口范围，可能导致识别失败")
        
        print(f"\n设置完成!")
        print(f"窗口内相对坐标: ({rel_x1}, {rel_y1}) -> ({rel_x2}, {rel_y2})")
        
        # 更新配置文件中的关卡识别区域
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # 使用正则表达式替换LEVEL_RECOGNITION_REGION
            new_region = f"'left': {rel_x1},  # 手动设置的坐标\n        'top': {rel_y1},   # 手动设置的坐标\n        'right': {rel_x2}, # 手动设置的坐标\n        'bottom': {rel_y2} # 手动设置的坐标"
            
            # 替换配置文件中的识别区域
            updated_content = re.sub(
                r"LEVEL_RECOGNITION_REGION = \{[^\}]*\}",
                f"LEVEL_RECOGNITION_REGION = {{\n        {new_region}\n    }}",
                config_content,
                flags=re.DOTALL
            )
            
            # 写回配置文件
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✓ 关卡识别区域已更新到配置文件: {config_path}")
            
            # 测试新的识别区域
            print("\n正在使用新的识别区域进行测试...")
            # 刷新配置模块
            import importlib
            import config
            importlib.reload(config)
            
            # 重新创建识别器并测试
            test_recognizer = LevelRecognizer()
            test_recognizer.find_game_window()
            new_region_screen = test_recognizer.get_level_region()
            if new_region_screen:
                print(f"新的识别区域(屏幕坐标): {new_region_screen}")
                level, confidence = test_recognizer.capture_and_recognize(new_region_screen, save_debug=True)
                if level:
                    print(f"✓ 测试成功! 识别到关卡: 第{level}关 (置信度: {confidence:.2f})")
                else:
                    print("✗ 测试失败，未能识别到关卡，请考虑调整区域")
            
            return True
            
        except Exception as e:
            print(f"更新配置文件失败: {e}")
            return False
        
    except Exception as e:
        print(f"设置过程出错: {e}")
        return False

def main():
    """主函数"""
    print("请选择操作模式:")
    print("1. 单次识别测试")
    print("2. 实时持续识别")
    print("3. 手动框选识别区域")
    
    choice = input("请输入选择 (1/2/3): ")
    
    if choice == '1':
        test_single_recognition()
    elif choice == '2':
        recognizer = LevelRecognizer()
        recognizer.recognize_level_realtime(interval=0.5)
    elif choice == '3':
        select_region_manually()
    else:
        print("无效的选择")
    
    print("\n程序已结束")
    input("按Enter键退出...")


if __name__ == "__main__":
    main()
