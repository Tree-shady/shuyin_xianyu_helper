import json
import re
import sys
import time
import random
import warnings
import easyocr
import win32gui
import win32con
import pyautogui
import cv2
import numpy as np
import argparse
from datetime import datetime
from loguru import logger
from PIL import ImageGrab, Image
from utils import GameHelper, stri_similar, retry_operation, check_game_version, load_version_history, save_version_history, track_usage_statistics
from config import Config
# 导入关卡识别器
from level_recognition import LevelRecognizer

# 忽略特定的PyTorch警告
warnings.filterwarnings("ignore", category=UserWarning, message="'pin_memory' argument is set as true but no accelerator is found")


class XianYuHelper:
    """
    咸鱼之王游戏辅助类
    专注于游戏特定逻辑，通用功能委托给GameHelper
    """
    def __init__(self, ocr_reader):
        """初始化"""
        # 初始化游戏辅助工具
        self.game_helper = GameHelper(ocr_reader, window_name=Config.GAME_WINDOW_TITLE)
        self.hwnd = self.game_helper.hwnd

        # 读取题库
        self.ans = []
        self.read_ans(Config.ANSWER_DATABASE_PATH)

        # 配置参数
        self.min_similarity = Config.MIN_SIMILARITY  # 最低相似度阈值
        self.color_thresholds = Config.COLOR_THRESHOLDS  # 颜色阈值
        
        # 初始化关卡识别器
        self.level_recognizer = None
        try:
            # 传入OCR引擎实例，避免重复初始化
            self.level_recognizer = LevelRecognizer(external_reader=ocr_reader)
            logger.info("关卡识别器初始化成功")
        except Exception as e:
            logger.warning(f"关卡识别器初始化失败: {str(e)}")
            
        logger.info("咸鱼之王辅助初始化完成")

    def read_ans(self, file_name):
        """读取答案题库"""
        try:
            with open(file_name, 'r', encoding="utf-8") as f:
                content = f.read()
                content = content.split("\n")

                for item in content:
                    try:
                        if not item.strip():
                            continue
                            
                        js = json.loads(item)

                        # 处理答案格式
                        if js['ans'] == 'A':
                            js['ans'] = js['a'][0]
                        elif js['ans'] == 'B':
                            js['ans'] = js['a'][1]

                        self.ans.append(js)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON解析错误: {e}, 行内容: {item}")
                    except KeyError as e:
                        logger.debug(f"键错误: {e}, 行内容: {item}")
                    except Exception as e:
                        logger.debug(f"处理行时出错: {e}, 行内容: {item}")

            logger.info("读取题库：{}", len(self.ans))
        except FileNotFoundError:
            logger.error(f"题库文件未找到: {file_name}")
            raise
        except Exception as e:
            logger.error(f"读取题库时出错: {e}")
            raise

            logger.info("读取题库：{}", len(self.ans))
            
    def recognize_bottom_buttons(self):
        """识别底部五个按钮（咸将、装备、战斗、宝箱、副本）
        
        Returns:
            list: 识别到的按钮名称列表，按从左到右顺序排列
        """
        logger.info("开始识别底部按钮")
        recognized_buttons = []
        
        try:
            # 检查游戏窗口是否存在且可见
            if not self.game_helper.check_window():
                logger.error("无法找到游戏窗口或窗口不可见")
                # 尝试使用备用方法 - 直接截取屏幕底部区域
                logger.info("尝试使用备用方法：截取屏幕底部区域")
                
                # 直接使用屏幕坐标而不依赖游戏窗口位置
                # 这是一个后备方案，当无法找到游戏窗口时使用
                screen_region = (0, 700, 800, 1500)  # 直接在屏幕上截取底部区域
                logger.info(f"使用备用屏幕区域: {screen_region}")
                
                # 截取屏幕底部区域
                img = self.game_helper.grab_screen_region(screen_region)
                if img is None:
                    logger.error("截取屏幕底部区域失败")
                    return []
                
                # 使用备用逻辑处理图像
                return self._process_button_image(img, use_window_offset=False)
            
            # 记录窗口大小信息
            logger.info(f"游戏窗口信息: 位置(left={self.game_helper.left}, top={self.game_helper.top}), "
                        f"大小(width={self.game_helper.right - self.game_helper.left}, "
                        f"height={self.game_helper.bottom - self.game_helper.top})")
            
            # 获取底部按钮区域
            if not hasattr(Config, 'BOTTOM_BUTTONS_REGION'):
                logger.error("配置中未找到底部按钮区域")
                return []
                
            button_region = Config.BOTTOM_BUTTONS_REGION
            
            # 计算实际的区域坐标（考虑游戏窗口的位置）
            # 确保识别区域不超出屏幕范围
            left = max(0, self.game_helper.left + button_region['left'])
            top = max(0, self.game_helper.top + button_region['top'])
            # 右侧和底部坐标可以适当超出，系统会自动裁剪到屏幕范围
            right = self.game_helper.left + button_region['right']
            bottom = self.game_helper.top + button_region['bottom']
            
            region_tuple = (left, top, right, bottom)
            
            logger.info(f"底部按钮识别区域: {region_tuple}")
            
            # 截取整个游戏窗口作为参考
            full_window_region = (
                self.game_helper.left, 
                self.game_helper.top, 
                self.game_helper.right, 
                self.game_helper.bottom
            )
            full_window_img = self.game_helper.grab_screen_region(full_window_region)
            
            if full_window_img:
                # 保存完整窗口截图用于调试
                debug_dir = "debug_screenshots"
                import os
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_debug_path = os.path.join(debug_dir, f"full_window_{timestamp}.png")
                full_window_img.save(full_debug_path)
                logger.info(f"已保存完整游戏窗口截图到: {full_debug_path}")
            
            # 截取底部按钮区域图像
            img = self.game_helper.grab_screen_region(region_tuple)
            if img is None:
                logger.error("截取底部按钮区域失败")
                return []
                
            # 处理按钮图像
            return self._process_button_image(img, use_window_offset=True)
            
            # 按x坐标排序，从左到右
            button_texts_with_positions.sort(key=lambda x: x[0])
            
            # 提取排序后的按钮文本
            sorted_buttons = [text for _, text, _ in button_texts_with_positions]
            logger.info(f"排序后的识别文本: {sorted_buttons}")
            
            # 与预期按钮名称匹配
            expected_buttons = Config.BOTTOM_BUTTONS_NAMES if hasattr(Config, 'BOTTOM_BUTTONS_NAMES') else []
            logger.info(f"预期的按钮名称: {expected_buttons}")
            
            # 对每个预期按钮，找到最匹配的识别结果
            for expected_btn in expected_buttons:
                best_match = None
                best_similarity = 0
                
                for btn_text in sorted_buttons:
                    similarity = stri_similar(btn_text, expected_btn)
                    logger.debug(f"比较 '{btn_text}' 和 '{expected_btn}'，相似度: {similarity:.2f}")
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = btn_text
                
                # 使用更低的相似度阈值以提高识别率
                if best_match and best_similarity > 0.25:  
                    recognized_buttons.append(expected_btn)
                    logger.info(f"匹配按钮: '{best_match}' -> '{expected_btn}' (相似度: {best_similarity:.2f})")
                else:
                    # 如果没有找到匹配，尝试使用部分匹配
                    if best_match:
                        logger.warning(f"相似度不足: '{best_match}' -> '{expected_btn}' (相似度: {best_similarity:.2f})")
                    else:
                        logger.warning(f"未识别到按钮: '{expected_btn}'")
                    
                    # 尝试字符级匹配
                    matched = False
                    for btn_text in sorted_buttons:
                        # 检查是否有共同字符
                        common_chars = set(btn_text) & set(expected_btn)
                        if len(common_chars) >= 1:  # 至少有一个共同字符
                            recognized_buttons.append(expected_btn)
                            logger.info(f"字符级匹配: '{btn_text}' -> '{expected_btn}' (共同字符: {common_chars})")
                            matched = True
                            break
                    
                    # 如果字符级匹配也失败，尝试更宽松的匹配策略
                    if not matched:
                        # 直接根据位置推断按钮
                        button_index = expected_buttons.index(expected_btn)
                        # 根据预期位置范围匹配
                        if button_texts_with_positions:
                            # 简化的位置匹配逻辑
                            expected_pos_range = {
                                0: (0, 150),    # 咸将 - 左侧
                                1: (150, 300),  # 装备 - 左中
                                2: (300, 450),  # 战斗 - 中间
                                3: (450, 600),  # 宝箱 - 右中
                                4: (600, 800)   # 副本 - 右侧
                            }
                            
                            if button_index in expected_pos_range:
                                min_x, max_x = expected_pos_range[button_index]
                                # 查找此位置范围内的文本
                                for x_pos, text, conf in button_texts_with_positions:
                                    if min_x <= x_pos <= max_x:
                                        recognized_buttons.append(expected_btn)
                                        logger.info(f"位置匹配: 在范围 {min_x}-{max_x} 找到文本 '{text}'，推断为 '{expected_btn}'")
                                        matched = True
                                        break
                    
                    if not matched:
                        recognized_buttons.append(None)  # 确实未识别到
            
            # 如果没有使用预期按钮进行匹配，则直接返回排序后的识别结果
            if not expected_buttons:
                recognized_buttons = sorted_buttons
            
            logger.info(f"底部按钮识别完成，识别到 {len([b for b in recognized_buttons if b])}/{len(recognized_buttons)} 个按钮")
            return recognized_buttons
            
        except Exception as e:
            logger.error(f"识别底部按钮时出错: {e}")
            import traceback
            logger.debug(f"错误堆栈: {traceback.format_exc()}")
            return []
    
    def _process_button_image(self, img, use_window_offset=True):
        """处理按钮图像并识别文本
        
        Args:
            img: 要处理的图像
            use_window_offset: 是否使用窗口偏移
            
        Returns:
            list: 识别到的按钮名称列表
        """
        recognized_buttons = []
        
        # 保存截图用于调试
        debug_dir = "debug_screenshots"
        import os
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_img_path = os.path.join(debug_dir, f"bottom_buttons_{timestamp}.png")
        img.save(debug_img_path)
        logger.info(f"已保存底部按钮区域截图到: {debug_img_path}")
        
        # 转换为numpy数组进行处理
        img_np = np.array(img)
        
        # 保存多种预处理结果以比较效果
        processed_images = {
            "original": img_np,
        }
        
        # 预处理1: 转换为灰度图
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        processed_images["grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # 预处理2: 自适应阈值（更强的对比度）
        thresh1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 3  # 调整参数增强对比度
        )
        processed_images["adaptive_thresh1"] = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB)
        
        # 预处理3: 不同参数的自适应阈值
        thresh2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 4  # 更大的块大小和不同的常量
        )
        processed_images["adaptive_thresh2"] = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2RGB)
        
        # 预处理4: 简单的二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        processed_images["binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # 保存所有预处理结果
        for name, proc_img in processed_images.items():
            if name != "original":  # 原始图像已保存
                proc_img_path = os.path.join(debug_dir, f"bottom_buttons_{timestamp}_{name}.png")
                Image.fromarray(proc_img).save(proc_img_path)
                logger.info(f"已保存预处理图像 '{name}' 到: {proc_img_path}")
        
        # 使用多种预处理图像进行OCR，合并结果
        all_ocr_results = []
        for name, proc_img in processed_images.items():
            logger.info(f"使用预处理图像 '{name}' 进行OCR识别...")
            result = self.game_helper.get_ocr_result(proc_img, use_cache=False)
            logger.info(f"预处理图像 '{name}' 的OCR结果: {result}")
            
            # 去重并添加到总结果
            for detection in result:
                # 检查是否已存在相似的文本
                text = detection[1].strip()
                confidence = detection[2] if len(detection) > 2 else 0
                
                # 只添加置信度较高的结果
                if text and confidence > 0.1:
                    all_ocr_results.append(detection)
        
        # 去重处理
        unique_results = []
        seen_texts = set()
        for detection in all_ocr_results:
            text = detection[1].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(detection)
        
        logger.info(f"合并去重后的OCR结果: {unique_results}")
        
        # 提取识别到的文本和位置信息
        button_texts_with_positions = []
        for detection in unique_results:
            try:
                # 获取文本和置信度
                text = detection[1].strip()
                confidence = detection[2] if len(detection) > 2 else 0
                
                # 获取文本的位置信息（使用左上角x坐标来确定左右顺序）
                top_left = detection[0][0]
                x_position = top_left[0]
                
                # 记录所有可能的文本
                if text:
                    button_texts_with_positions.append((x_position, text, confidence))
                    logger.info(f"识别到文本: '{text}'，位置: {x_position}，置信度: {confidence:.2f}")
            except Exception as e:
                logger.debug(f"处理OCR结果时出错: {e}")
                continue
            
        # 按x坐标排序，从左到右
        button_texts_with_positions.sort(key=lambda x: x[0])
        
        # 提取排序后的按钮文本
        sorted_buttons = [text for _, text, _ in button_texts_with_positions]
        logger.info(f"排序后的识别文本: {sorted_buttons}")
        
        # 与预期按钮名称匹配
        expected_buttons = Config.BOTTOM_BUTTONS_NAMES if hasattr(Config, 'BOTTOM_BUTTONS_NAMES') else []
        logger.info(f"预期的按钮名称: {expected_buttons}")
        
        # 对每个预期按钮，找到最匹配的识别结果
        for expected_btn in expected_buttons:
            best_match = None
            best_similarity = 0
            
            for btn_text in sorted_buttons:
                similarity = stri_similar(btn_text, expected_btn)
                logger.debug(f"比较 '{btn_text}' 和 '{expected_btn}'，相似度: {similarity:.2f}")
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = btn_text
            
            # 使用更低的相似度阈值以提高识别率
            if best_match and best_similarity > 0.25:  
                recognized_buttons.append(expected_btn)
                logger.info(f"匹配按钮: '{best_match}' -> '{expected_btn}' (相似度: {best_similarity:.2f})")
            else:
                # 如果没有找到匹配，尝试使用部分匹配
                if best_match:
                    logger.warning(f"相似度不足: '{best_match}' -> '{expected_btn}' (相似度: {best_similarity:.2f})")
                else:
                    logger.warning(f"未识别到按钮: '{expected_btn}'")
                
                # 尝试字符级匹配
                matched = False
                for btn_text in sorted_buttons:
                    # 检查是否有共同字符
                    common_chars = set(btn_text) & set(expected_btn)
                    if len(common_chars) >= 1:  # 至少有一个共同字符
                        recognized_buttons.append(expected_btn)
                        logger.info(f"字符级匹配: '{btn_text}' -> '{expected_btn}' (共同字符: {common_chars})")
                        matched = True
                        break
                
                # 如果字符级匹配也失败，尝试更宽松的匹配策略
                if not matched:
                    # 直接根据位置推断按钮
                    button_index = expected_buttons.index(expected_btn)
                    # 根据预期位置范围匹配
                    if button_texts_with_positions:
                        # 简化的位置匹配逻辑
                        expected_pos_range = {
                            0: (0, 150),    # 咸将 - 左侧
                            1: (150, 300),  # 装备 - 左中
                            2: (300, 450),  # 战斗 - 中间
                            3: (450, 600),  # 宝箱 - 右中
                            4: (600, 800)   # 副本 - 右侧
                        }
                        
                        if button_index in expected_pos_range:
                            min_x, max_x = expected_pos_range[button_index]
                            # 查找此位置范围内的文本
                            for x_pos, text, conf in button_texts_with_positions:
                                if min_x <= x_pos <= max_x:
                                    recognized_buttons.append(expected_btn)
                                    logger.info(f"位置匹配: 在范围 {min_x}-{max_x} 找到文本 '{text}'，推断为 '{expected_btn}'")
                                    matched = True
                                    break
                
                if not matched:
                    recognized_buttons.append(None)  # 确实未识别到
        
        # 如果没有使用预期按钮进行匹配，则直接返回排序后的识别结果
        if not expected_buttons:
            recognized_buttons = sorted_buttons
        
        logger.info(f"底部按钮识别完成，识别到 {len([b for b in recognized_buttons if b])}/{len(recognized_buttons)} 个按钮")
        return recognized_buttons
            
    def click_bottom_button(self, button_name):
        """点击指定的底部按钮
        
        Args:
            button_name: 按钮名称，必须是以下之一：'咸将', '装备', '战斗', '宝箱', '副本'
            
        Returns:
            bool: 点击是否成功
        """
        import random
        import time
        
        logger.info(f"尝试点击底部按钮: {button_name}")
        
        try:
            # 检查按钮名称是否有效
            valid_buttons = Config.BOTTOM_BUTTONS_NAMES if hasattr(Config, 'BOTTOM_BUTTONS_NAMES') else []
            if button_name not in valid_buttons:
                logger.error(f"无效的按钮名称: {button_name}，有效名称: {valid_buttons}")
                return False
            
            # 检查游戏窗口是否可用
            window_available = hasattr(self.game_helper, 'left') and hasattr(self.game_helper, 'top')
            window_available = window_available and self.game_helper.left is not None and self.game_helper.top is not None
            
            # 尝试获取按钮坐标
            if hasattr(Config, 'BOTTOM_BUTTONS_POSITIONS') and button_name in Config.BOTTOM_BUTTONS_POSITIONS:
                # 获取配置中的相对坐标
                rel_x, rel_y = Config.BOTTOM_BUTTONS_POSITIONS[button_name]
                
                if window_available:
                    # 窗口可用，使用窗口坐标计算绝对位置
                    screen_x = self.game_helper.left + rel_x
                    screen_y = self.game_helper.top + rel_y
                    
                    logger.info(f"游戏窗口坐标: (left={self.game_helper.left}, top={self.game_helper.top})")
                    logger.info(f"按钮 '{button_name}' 相对坐标: ({rel_x}, {rel_y})")
                    logger.info(f"按钮 '{button_name}' 实际屏幕坐标: ({screen_x}, {screen_y})")
                else:
                    # 窗口不可用，使用备用策略 - 尝试直接使用相对坐标作为屏幕坐标
                    # 这是一个后备方案，假设游戏窗口在屏幕左上角或已经最大化
                    logger.warning("无法获取游戏窗口位置信息，使用备用点击策略")
                    
                    # 使用直接的屏幕坐标估算（基于按钮在底部栏的位置）
                    # 这些值可以根据实际游戏界面进行调整
                    # 优化的备用坐标，增加随机性以模拟人类点击
                    fallback_y = 970  # 优化的底部按钮Y坐标
                    fallback_x_values = {
                        '咸将': 100,   # 优化的坐标
                        '装备': 260,   # 优化的装备按钮坐标
                        '战斗': 420,   # 优化的坐标
                        '宝箱': 580,   # 优化的坐标
                        '副本': 740    # 优化的坐标
                    }
                    
                    if button_name in fallback_x_values:
                        # 基础坐标 + 随机偏移，模拟人类行为
                        base_x = fallback_x_values[button_name]
                        base_y = fallback_y
                        
                        # 添加±10像素的随机偏移
                        random_offset_x = random.randint(-10, 10)
                        random_offset_y = random.randint(-5, 5)
                        
                        screen_x = base_x + random_offset_x
                        screen_y = base_y + random_offset_y
                        
                        logger.info(f"使用备用策略的按钮坐标: ({base_x}, {base_y}) + 随机偏移 ({random_offset_x}, {random_offset_y}) = ({screen_x}, {screen_y})")
                    else:
                        logger.error(f"无法确定按钮 '{button_name}' 的备用坐标")
                        return False
                
                # 执行点击操作 - 添加改进的重试机制
                max_attempts = 3  # 增加重试次数
                success = False
                
                for attempt in range(max_attempts):
                    # 每次尝试都添加小的随机偏移
                    final_x = screen_x + random.randint(-3, 3)
                    final_y = screen_y + random.randint(-2, 2)
                    
                    logger.info(f"尝试点击按钮 {button_name} (尝试 {attempt + 1}/{max_attempts})，坐标: ({final_x}, {final_y})")
                    
                    try:
                        if window_available:
                            # 窗口可用时，使用窗口相对坐标点击
                            # 计算相对坐标
                            rel_click_x = final_x - self.game_helper.left
                            rel_click_y = final_y - self.game_helper.top
                            
                            # 记录点击类型
                            logger.debug(f"使用窗口相对坐标点击: ({rel_click_x}, {rel_click_y})")
                            
                            # 尝试使用不同的点击方式
                            success = self.game_helper.left_click(final_x, final_y, delay=0.5)
                            
                            # 如果失败，尝试直接使用SendMessage方式
                            if not success and hasattr(self.game_helper, 'hwnd') and self.game_helper.hwnd:
                                logger.debug("尝试使用SendMessage方式点击")
                                import windows
                                try:
                                    windows.left_click_position(self.game_helper.hwnd, final_x, final_y, sleep_time=0.5)
                                    success = True
                                except Exception as e:
                                    logger.debug(f"SendMessage点击失败: {e}")
                        else:
                            # 窗口不可用时，使用绝对屏幕坐标直接点击
                            logger.info(f"使用绝对屏幕坐标点击: ({final_x}, {final_y})")
                            import windows
                            
                            # 尝试不同的点击方法
                            try:
                                # 方法1: 直接调用windows模块的left_click_position
                                windows.left_click_position(None, final_x, final_y, sleep_time=0.5)
                                success = True
                            except Exception as e1:
                                logger.warning(f"方法1点击失败: {e1}")
                                # 方法2: 尝试使用pyautogui作为备用
                                try:
                                    import pyautogui
                                    # 移动鼠标到目标位置
                                    pyautogui.moveTo(final_x, final_y, duration=0.2 + random.random() * 0.3)  # 添加随机移动时间
                                    # 点击
                                    pyautogui.click()
                                    logger.info("使用pyautogui备用方法点击成功")
                                    success = True
                                except Exception as e2:
                                    logger.error(f"方法2点击失败: {e2}")
                                    success = False
                    except Exception as e:
                        logger.error(f"点击过程中发生异常: {e}")
                        success = False
                    
                    if success:
                        logger.info(f"第 {attempt + 1} 次尝试成功点击按钮: {button_name}")
                        break
                    else:
                        logger.warning(f"第 {attempt + 1} 次尝试点击按钮 {button_name} 失败，准备重试...")
                        # 每次重试使用不同的偏移量
                        retry_offset_x = random.randint(15, 30) * (-1 if attempt % 2 == 0 else 1)
                        retry_offset_y = random.randint(5, 15)
                        
                        screen_x += retry_offset_x
                        screen_y += retry_offset_y
                        logger.info(f"调整重试坐标为: ({screen_x}, {screen_y}) + 下一次随机偏移")
                        
                        # 增加等待时间，让游戏有时间响应
                        wait_time = 0.2 + random.random() * 0.3
                        time.sleep(wait_time)
                
                if success:
                    logger.info(f"成功点击按钮: {button_name}")
                    # 添加随机延迟让游戏有时间响应点击，模拟人类行为
                    response_time = 0.3 + random.random() * 0.5
                    logger.debug(f"等待游戏响应，延迟 {response_time:.2f} 秒")
                    time.sleep(response_time)
                    return True
                else:
                    logger.error(f"所有尝试都失败，无法点击按钮: {button_name}")
                    # 添加调试信息，帮助用户诊断问题
                    logger.debug("点击失败可能原因:")
                    logger.debug("1. 游戏窗口可能被遮挡或不在前台")
                    logger.debug("2. 按钮坐标可能需要调整")
                    logger.debug("3. 游戏可能有反作弊机制拦截了点击")
                    logger.debug("4. 请检查游戏是否正常运行，窗口标题是否正确")
                    return False
            else:
                logger.error(f"未在配置中找到按钮 '{button_name}' 的位置信息")
                return False
        except Exception as e:
            logger.error(f"处理点击按钮 '{button_name}' 时发生错误: {str(e)}")
            return False
            
            # 如果没有BOTTOM_BUTTONS_POSITIONS配置，尝试使用BOTTOM_BUTTONS_CLICK_COORDS作为备选
            if hasattr(Config, 'BOTTOM_BUTTONS_CLICK_COORDS') and hasattr(Config, 'BOTTOM_BUTTONS_NAMES'):
                try:
                    button_index = Config.BOTTOM_BUTTONS_NAMES.index(button_name)
                    if 0 <= button_index < len(Config.BOTTOM_BUTTONS_CLICK_COORDS):
                        click_x, click_y = Config.BOTTOM_BUTTONS_CLICK_COORDS[button_index]
                        
                        if window_available:
                            screen_x = self.game_helper.left + click_x
                            screen_y = self.game_helper.top + click_y
                        else:
                            # 使用备用坐标
                            screen_x = click_x
                            screen_y = click_y
                            logger.warning("使用备用坐标系统，假设游戏窗口在标准位置")
                        
                        logger.info(f"使用备用坐标配置: ({screen_x}, {screen_y})")
                        success = self.game_helper.left_click(screen_x, screen_y, delay=0.5)
                        
                        if success:
                            logger.info(f"成功使用备用坐标点击按钮: {button_name}")
                            return True
                except (ValueError, IndexError):
                    logger.debug("备用坐标计算失败")
            
            logger.error(f"未找到按钮 '{button_name}' 的有效坐标配置")
            return False
                
        except Exception as e:
            logger.error(f"点击底部按钮时出错: {e}")
            import traceback
            logger.debug(f"错误堆栈: {traceback.format_exc()}")
            return False

    def get_task(self):
        """识别当前任务"""

    def task_auto_answer(self, duration=None):
        """自动答题任务
        
        Args:
            duration: 任务持续时间（秒），None表示无限时间
        """
        logger.info("开始自动答题任务")
        question_str_old = ""
        
        # 设置任务开始时间
        task_start_time = datetime.now()
        
        try:
            while True:
                # 检查任务是否达到设定的持续时间
                if duration is not None and duration > 0:
                    elapsed_time = (datetime.now() - task_start_time).seconds
                    if elapsed_time >= duration:
                        logger.info(f"任务已达到设定的持续时间({duration}秒)，自动结束")
                        break
                        
                # 检查窗口状态
                if not self.game_helper.check_window():
                    logger.warning("窗口异常，等待恢复...")
                    time.sleep(Config.WINDOW_RECOVERY_DELAY)
                    continue
                    
                try:
                    # 题目区域
                    question_region = (
                        self.game_helper.left + Config.QUESTION_REGION_LEFT,
                        self.game_helper.top + Config.QUESTION_REGION_TOP,
                        self.game_helper.left + Config.QUESTION_REGION_LEFT + Config.QUESTION_REGION_WIDTH,
                        self.game_helper.top + Config.QUESTION_REGION_TOP + Config.QUESTION_REGION_HEIGHT
                    )
                    
                    # 获取原始图像
                    img = self.game_helper.grab_screen_region(question_region)
                    
                    # 直接使用OpenCV批量处理，避免PIL和numpy之间的多次转换
                    # 转换为numpy数组
                    img_np = np.array(img)
                    
                    # 创建掩码进行批量颜色处理
                    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                    # 从配置中获取颜色阈值
                    if isinstance(Config.COLOR_THRESHOLDS, dict):
                        # 如果是字典格式（从config.py中读取）
                        # 处理黑色文本
                        lower_black = np.array(Config.COLOR_THRESHOLDS['lower_black'])
                        upper_black = np.array(Config.COLOR_THRESHOLDS['upper_black'])
                        black_mask = cv2.inRange(cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV), lower_black, upper_black)
                        
                        # 处理白色文本（在深色背景上）
                        lower_white = np.array(Config.COLOR_THRESHOLDS['lower_white'])
                        upper_white = np.array(Config.COLOR_THRESHOLDS['upper_white'])
                        white_mask = cv2.inRange(cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV), lower_white, upper_white)
                        
                        # 合并掩码
                        mask = cv2.bitwise_or(black_mask, white_mask)
                    else:
                        # 兼容旧格式的颜色阈值
                        if self.color_thresholds is None:
                            # 默认颜色阈值
                            color_thresholds = [[48, 48, 48, 32], [82, 31, 26, 40]]
                        else:
                            color_thresholds = self.color_thresholds
                        
                        # 批量处理每个颜色阈值
                        for color in color_thresholds:
                            lower = np.array([color[0]-color[3], color[1]-color[3], color[2]-color[3]])
                            upper = np.array([color[0]+color[3], color[1]+color[3], color[2]+color[3]])
                            color_mask = cv2.inRange(img_np, lower, upper)
                            mask = cv2.bitwise_or(mask, color_mask)
                    
                    # 批量二值化处理
                    processed_img = np.zeros_like(img_np)
                    processed_img[mask > 0] = [0, 0, 0]  # 黑色（文本）
                    processed_img[mask == 0] = [255, 255, 255]  # 白色（背景）
                    
                    # 应用额外的图像增强（可选）
                    # 高斯模糊降噪
                    processed_img = cv2.GaussianBlur(processed_img, (1, 1), 0)
                    
                    # OCR识别（带缓存）
                    result = self.game_helper.get_ocr_result(processed_img)
                    
                    # 拼接识别结果
                    question_str = ""
                    for item in result:
                        question_str += item[1]
                    
                    # 去除空格
                    question_str = question_str.replace(" ", "")
                    
                    # 题目变化时才处理
                    if question_str and question_str != question_str_old:
                        question_str_old = question_str
                        
                        # 查找最佳匹配答案（使用优化的查找算法）
                        best_match = {}
                        best_similarity = 0
                        
                        # 首先进行快速过滤，只计算相似度高于一定阈值的项
                        # 这可以减少计算量，特别是在题库很大的情况下
                        for item in self.ans:
                            # 简单的字符串包含检查作为快速过滤
                            if any(keyword in question_str for keyword in item['q'][:10]) or \
                               any(keyword in item['q'] for keyword in question_str[:10]):
                                similarity = stri_similar(question_str, item['q'])
                                if similarity > best_similarity:
                                    best_match = item
                                    best_similarity = similarity
                        
                        logger.debug("题目识别结果：{}, 相似度：{:.2f}", question_str, best_similarity)
                        
                        if best_match and best_similarity > self.min_similarity:
                            logger.info("题库对比：{}，答案：{}", best_match['q'], best_match['ans'])
                            
                            # 根据答案点击对应按钮
                            if best_match['ans'] == '对':
                                self.game_helper.left_click(166, 930, delay=0.02)
                            elif best_match['ans'] == '错':
                                self.game_helper.left_click(422, 923, delay=0.02)
                except Exception as e:
                    logger.error(f"答题过程中出错: {e}")
                    time.sleep(Config.ERROR_RETRY_DELAY)
                
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("自动答题任务已停止")
        except Exception as e:
            logger.critical(f"自动答题任务异常终止: {e}")
            track_usage_statistics("auto_answer", success=False)

    def task_auto_pass(self, duration=None):
        """自动推图任务
        
        Args:
            duration: 任务持续时间（秒），None表示无限时间
        """
        logger.info("===== 自动推图任务启动 =====")
        logger.debug(f"任务持续时间设置: {'无限' if duration is None else f'{duration}秒'}")
        
        try:
            # 初始化关卡识别区域
            level_region = Config.LEVEL_RECOGNITION_REGION.copy()
            logger.info(f"关卡识别区域已设置: 左上({level_region['left']}, {level_region['top']}) - 右下({level_region['right']}, {level_region['bottom']})")
            
            # 如果关卡识别器可用，尝试预加载
            if self.level_recognizer:
                # 尝试查找游戏窗口
                window_found = self.level_recognizer.find_game_window()
                if window_found:
                    logger.info("关卡识别器成功找到游戏窗口")
                else:
                    logger.warning("关卡识别器未能找到游戏窗口，将使用备用方法")
            else:
                logger.warning("关卡识别器未初始化，将使用备用识别方法")
            
            # 设置任务开始时间和持续时间
            task_start_time = datetime.now()
            loop_count = 0
            click_count = 0
            game_level = "未知"
            last_successful_level = "未知"
            
            # 初始化缓存属性
            self._last_recognition_time = 0
            self._last_recognized_level = ""
            
            while True:
                loop_count += 1
                current_time = time.time()
                
                # 定期显示运行状态信息
                if loop_count % 100 == 0:
                    elapsed_time = (datetime.now() - task_start_time).seconds
                    logger.info(f"运行统计 - 循环次数: {loop_count}, 点击次数: {click_count}, 运行时间: {elapsed_time}秒, 当前关卡: {game_level}")
                
                # 执行关卡识别的条件：
                # 1. 第一次循环
                # 2. 距离上次成功识别超过120秒
                if (self._last_recognition_time == 0) or (current_time - self._last_recognition_time > 120):
                    try:
                        # 调用新的无参数关卡识别方法
                        recognized_level = self._recognize_level()
                        if recognized_level:
                            new_game_level = f"第{recognized_level}关"
                            # 检查关卡是否更新
                            if new_game_level != game_level:
                                game_level = new_game_level
                                last_successful_level = game_level
                                logger.info(f"识别到关卡: {game_level}，120秒内将不再进行识别以节省资源")
                            else:
                                logger.debug(f"关卡未更新: {game_level}")
                            
                            self._last_recognition_time = current_time  # 更新上次识别时间
                        else:
                            logger.debug(f"未识别到关卡，保持上次识别结果: {last_successful_level}")
                    except Exception as e:
                        logger.warning(f"关卡识别过程中出错: {str(e)}")
                
                # 检查任务是否达到设定的持续时间
                if duration is not None and duration > 0:
                    elapsed_time = (datetime.now() - task_start_time).seconds
                    if elapsed_time >= duration:
                        logger.info(f"任务已达到设定的持续时间({duration}秒)，自动结束")
                        logger.info(f"任务执行统计 - 总循环次数: {loop_count}, 实际运行时间: {elapsed_time}秒")
                        break
                
                # 检查窗口状态
                if not self.game_helper.check_window():
                    logger.warning("窗口异常，等待恢复...")
                    time.sleep(Config.WINDOW_RECOVERY_DELAY)
                    continue
                
                try:
                    # 获取推图点击坐标，确保不会点击到底部按钮区域（Y < 700）
                    battle_x = Config.PASS_LEVEL_CLICKS['battle'][0]
                    # 确保Y坐标在底部按钮区域上方（Y < 700）
                    battle_y = min(Config.PASS_LEVEL_CLICKS['battle'][1], 650)
                    
                    logger.debug(f"循环 {loop_count}: 执行战斗点击 - 安全坐标({battle_x}, {battle_y}) - 已确保在底部按钮区域上方")
                    self.game_helper.left_click(battle_x, battle_y, delay=0.01)
                    click_count += 1
                    
                except Exception as e:
                    logger.error(f"推图过程中出错: {str(e)}")
                    time.sleep(Config.ERROR_RETRY_DELAY)
                    
        except KeyboardInterrupt:
            logger.info("自动推图任务被用户中断")
            logger.info(f"任务中断统计 - 循环次数: {loop_count}, 运行时间: {(datetime.now() - task_start_time).seconds}秒")
        except Exception as e:
            logger.critical(f"自动推图任务异常终止: {str(e)}")
            track_usage_statistics("auto_pass", success=False)
            # 尝试发送错误通知
            try:
                # 移除对未定义的feishu属性的引用
                # 替换为更简单的错误处理逻辑
                logger.debug(f"推图任务异常信息: 错误类型: {type(e).__name__}, 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 循环次数: {loop_count}")
            except Exception as notify_error:
                logger.debug(f"记录错误信息失败: {str(notify_error)}")
        finally:
            logger.info(f"===== 自动推图任务结束 =====")
            logger.info(f"最终统计 - 总循环次数: {loop_count}, 总运行时间: {(datetime.now() - task_start_time).seconds}秒")
    
    def _recognize_level(self):
        """识别当前关卡"""
        try:
            # 添加缓存机制，避免短时间内重复识别
            current_time = time.time()
            
            # 初始化缓存属性（如果不存在）
            if not hasattr(self, '_last_recognition_time'):
                self._last_recognition_time = 0
            if not hasattr(self, '_last_recognized_level'):
                self._last_recognized_level = ""
            
            # 检查缓存是否有效
            if current_time - self._last_recognition_time < 5.0:  # 5秒缓存
                logger.debug(f"使用缓存关卡识别结果: {self._last_recognized_level}")
                return self._last_recognized_level
            
            # 确保识别区域有效
            if not hasattr(Config, 'LEVEL_RECOGNITION_REGION') or not Config.LEVEL_RECOGNITION_REGION:
                logger.error("配置中未找到关卡识别区域")
                return ""
                
            logger.debug("使用备用识别方法识别关卡")
            recognized_number = self._recognize_level_fallback(Config.LEVEL_RECOGNITION_REGION)
            
            if recognized_number:
                # 更新缓存
                self._last_recognition_time = current_time
                self._last_recognized_level = recognized_number
                logger.info(f"关卡识别成功: {recognized_number}，已更新缓存")
            
            return recognized_number
            
        except Exception as e:
            logger.error(f"关卡识别整体出错: {str(e)}")
            import traceback
            logger.debug(f"错误堆栈: {traceback.format_exc()}")
            return ""
    
    def _recognize_level_fallback(self, region):
        """备用关卡识别方法，增强版
        
        Args:
            region: 关卡识别区域
            
        Returns:
            str: 识别到的关卡号，如果识别失败则返回空字符串
        """
        try:
            # 确保region格式正确
            if isinstance(region, dict):
                region_tuple = (region['left'], region['top'], region['right'], region['bottom'])
            else:
                region_tuple = region
                
            logger.debug(f"识别区域: {region_tuple}")
            
            # 使用game_helper的截图功能
            try:
                img = self.game_helper.grab_screen_region(region_tuple)
                if img is None or img.size[0] == 0 or img.size[1] == 0:
                    logger.error("截图失败：图像为空")
                    return ""
            except Exception as img_error:
                logger.error(f"截图失败: {img_error}")
                import traceback
                logger.debug(f"截图错误堆栈: {traceback.format_exc()}")
                return ""
            
            # 转换为numpy数组
            img_np = np.array(img)
            
            # 简化版图像预处理 - 只使用灰度转换
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            
            # 收集所有OCR结果
            all_number_candidates = []
            all_confidence_scores = []
            
            # 对原始图像进行OCR
            try:
                result_original = self.game_helper.get_ocr_result(img_np, use_cache=False)
                logger.debug(f"原始图像OCR结果: {result_original}")
                for item in result_original:
                    text = item[1].strip() if len(item) > 1 else ""
                    confidence = item[2] if len(item) > 2 else 0
                    
                    logger.debug(f"识别文本: '{text}'，置信度: {confidence:.2f}")
                    
                    # 提取数字并验证
                    digits = re.sub("\\D", "", text)
                    if digits and 1 <= len(digits) <= 4:  # 放宽数字长度限制
                        # 优先考虑包含"第"或"关"字的文本中的数字
                        if '第' in text or '关' in text:
                            confidence += 0.3  # 增加权重
                        logger.debug(f"[原始图像] 识别到数字: '{digits}'，置信度: {confidence:.2f}")
                        all_number_candidates.append(digits)
                        all_confidence_scores.append(confidence)
            except Exception as e:
                logger.debug(f"原始图像处理出错: {e}")
                logger.debug(f"错误堆栈: {traceback.format_exc()}")
            
            # 对灰度图像进行OCR
            try:
                result_gray = self.game_helper.get_ocr_result(gray, use_cache=False)
                logger.debug(f"灰度图像OCR结果: {result_gray}")
                for item in result_gray:
                    text = item[1].strip() if len(item) > 1 else ""
                    confidence = item[2] if len(item) > 2 else 0
                    
                    # 提取数字并验证
                    digits = re.sub("\\D", "", text)
                    if digits and 1 <= len(digits) <= 4:
                        if '第' in text or '关' in text:
                            confidence += 0.3  # 增加权重
                        logger.debug(f"[灰度图像] 识别到数字: '{digits}'，置信度: {confidence:.2f}")
                        all_number_candidates.append(digits)
                        all_confidence_scores.append(confidence)
            except Exception as e:
                logger.debug(f"灰度图像处理出错: {e}")
            
            # 统计和选择最佳结果
            if all_number_candidates:
                # 找出置信度最高的数字
                best_idx = all_confidence_scores.index(max(all_confidence_scores))
                best_number = all_number_candidates[best_idx]
                best_confidence = all_confidence_scores[best_idx]
                
                # 降低最低置信度要求以提高识别率
                if best_confidence > 0.1:
                    logger.info(f"[备用识别] 识别到关卡: {best_number} (置信度: {best_confidence:.2f})")
                    return best_number
                else:
                    logger.warning(f"[备用识别] 识别到数字但置信度过低: {best_number} (置信度: {best_confidence:.2f})")
            else:
                logger.warning("[备用识别] 未识别到有效数字")
                
            return ""
        
        except Exception as e:
            logger.error(f"备用关卡识别时出错: {e}")
            import traceback
            logger.debug(f"备用识别错误堆栈: {traceback.format_exc()}")
            return ""

    def task_auto_tower(self, duration=None):
        """自动推塔任务
        
        Args:
            duration: 任务持续时间（秒），None表示无限时间
        """
        logger.info("===== 自动推塔任务启动 =====")
        logger.debug(f"任务持续时间设置: {'无限' if duration is None else f'{duration}秒'}")
        
        # 获取奖励检测区域
        region = Config.TOWER_REWARD_REGION
        (x1, y1) = win32gui.ClientToScreen(self.hwnd, (region['left'], region['top']))
        (x2, y2) = win32gui.ClientToScreen(self.hwnd, (region['right'], region['bottom']))
        reward_region = (x1, y1, x2, y2)
        logger.debug(f"奖励检测区域: ({x1}, {y1}, {x2}, {y2})")
        
        loop_count = 0
        click_count = 0
        reward_claim_count = 0
        
        # 设置任务开始时间
        task_start_time = datetime.now()
        
        try:
            while True:
                # 检查任务是否达到设定的持续时间
                if duration is not None and duration > 0:
                    elapsed_time = (datetime.now() - task_start_time).seconds
                    if elapsed_time >= duration:
                        logger.info(f"任务已达到设定的持续时间({duration}秒)，自动结束")
                        logger.info(f"任务执行统计 - 总循环次数: {loop_count}, 总点击次数: {click_count}, 领取奖励次数: {reward_claim_count}, 实际运行时间: {elapsed_time}秒")
                        break
                        
                loop_count += 1
                
                # 定期显示运行状态信息
                if loop_count % 50 == 0:
                    elapsed_time = (datetime.now() - task_start_time).seconds
                    logger.info(f"运行统计 - 循环次数: {loop_count}, 点击次数: {click_count}, 领取奖励次数: {reward_claim_count}, 运行时间: {elapsed_time}秒")
                
                # 检查窗口状态
                if not self.game_helper.check_window():
                    logger.warning("窗口异常，等待恢复...")
                    time.sleep(Config.WINDOW_RECOVERY_DELAY)
                    continue
                
                try:
                    # 基础点击操作
                    logger.debug(f"循环 {loop_count}: 执行基础点击1 - 坐标({Config.TOWER_CLICKS['base1'][0]}, {Config.TOWER_CLICKS['base1'][1]})")
                    self.game_helper.left_click(Config.TOWER_CLICKS['base1'][0], 
                                              Config.TOWER_CLICKS['base1'][1], delay=0.01)
                    click_count += 1
                    
                    logger.debug(f"循环 {loop_count}: 执行基础点击2 - 坐标({Config.TOWER_CLICKS['base2'][0]}, {Config.TOWER_CLICKS['base2'][1]})")
                    self.game_helper.left_click(Config.TOWER_CLICKS['base2'][0], 
                                              Config.TOWER_CLICKS['base2'][1], delay=0.01)
                    click_count += 1
                    
                    # 周期性检查和特殊操作
                    if loop_count % Config.TOWER_LOOP_CHECK_INTERVAL == 0:
                        logger.info(f"周期性检查 (循环 {loop_count})")
                        logger.debug(f"循环 {loop_count}: 执行周期性检查点击 - 坐标({Config.TOWER_CLICKS['periodic'][0]}, {Config.TOWER_CLICKS['periodic'][1]})")
                        self.game_helper.left_click(Config.TOWER_CLICKS['periodic'][0], 
                                                  Config.TOWER_CLICKS['periodic'][1], delay=0.01)
                        click_count += 1
                           
                        # 检查奖励和鱼干状态
                        try:
                            logger.debug("检查奖励和鱼干状态...")
                            img = ImageGrab.grab(bbox=reward_region)
                            gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
                            result = self.game_helper.get_ocr_result(gray_img)
                            
                            has_reward = False
                            low_fish_dry = False
                            
                            for item in result:
                                text = item[1]
                                logger.debug(f"识别到文本: {text}")
                                
                                # 检查奖励领取
                                if '奖励' in text and '领取' in text:
                                    logger.info("检测到可领取奖励")
                                    has_reward = True
                                    break
                                
                                # 检查鱼干不足
                                if '鱼干' in text:
                                    logger.info("检测到鱼干不足提示")
                                    low_fish_dry = True
                                    break
                            
                            # 处理奖励领取
                            if has_reward:
                                logger.info("开始领取奖励")
                                logger.debug(f"循环 {loop_count}: 执行领取奖励1 - 坐标({Config.TOWER_CLICKS['claim_reward1'][0]}, {Config.TOWER_CLICKS['claim_reward1'][1]})")
                                self.game_helper.left_click(Config.TOWER_CLICKS['claim_reward1'][0], 
                                                          Config.TOWER_CLICKS['claim_reward1'][1], delay=2)
                                click_count += 1
                                
                                logger.debug(f"循环 {loop_count}: 执行领取奖励2 - 坐标({Config.TOWER_CLICKS['claim_reward2'][0]}, {Config.TOWER_CLICKS['claim_reward2'][1]})")
                                self.game_helper.left_click(Config.TOWER_CLICKS['claim_reward2'][0], 
                                                          Config.TOWER_CLICKS['claim_reward2'][1], delay=2)
                                click_count += 1
                                
                                reward_claim_count += 1
                                logger.info(f"已领取奖励，累计领取次数: {reward_claim_count}")
                            
                            # 处理鱼干不足
                            if low_fish_dry:
                                logger.info("鱼干不足，结束推塔任务")
                                return  # 优雅退出
                        except Exception as e:
                            logger.error(f"检查奖励状态时出错: {str(e)}")
                
                except Exception as e:
                    logger.error(f"推塔过程中出错: {str(e)}")
                    time.sleep(Config.ERROR_RETRY_DELAY)
                    
        except KeyboardInterrupt:
            logger.info("自动推塔任务被用户中断")
            logger.info(f"任务中断统计 - 循环次数: {loop_count}, 点击次数: {click_count}, 领取奖励次数: {reward_claim_count}, 运行时间: {(datetime.now() - task_start_time).seconds}秒")
        except Exception as e:
            logger.critical(f"自动推塔任务异常终止: {str(e)}")
            track_usage_statistics("auto_tower", success=False)
        finally:
            logger.info(f"===== 自动推塔任务结束 =====")
            logger.info(f"最终统计 - 总循环次数: {loop_count}, 总点击次数: {click_count}, 领取奖励次数: {reward_claim_count}, 总运行时间: {(datetime.now() - task_start_time).seconds}秒")

    def task_auto_fish(self, duration=None):
        """自动钓鱼任务
        
        Args:
            duration: 任务持续时间（秒），None表示无限时间
        """
        logger.info("开始自动钓鱼任务")
        
        # 设置任务开始时间
        task_start_time = datetime.now()
        
        try:
            while True:
                # 检查任务是否达到设定的持续时间
                if duration is not None and duration > 0:
                    elapsed_time = (datetime.now() - task_start_time).seconds
                    if elapsed_time >= duration:
                        logger.info(f"任务已达到设定的持续时间({duration}秒)，自动结束")
                        break
                        
                # 检查窗口状态
                if not self.game_helper.check_window():
                    logger.warning("窗口异常，等待恢复...")
                    time.sleep(Config.WINDOW_RECOVERY_DELAY)
                    continue
                
                try:
                    logger.info("执行钓鱼流程")
                    
                    # 执行钓鱼步骤
                    for i, (x, y, delay) in enumerate(Config.FISHING_STEPS):
                        self.game_helper.left_click(x, y, delay=delay)
                        
                        # 特殊处理：钓鱼动画等待
                        if i == 2:  # 开始钓鱼后的步骤
                            logger.info("等待钓鱼动画...")
                            time.sleep(Config.FISHING_ANIMATION_WAIT)
                    
                    # 等待收集动画完成
                    logger.info("等待收集动画...")
                    time.sleep(Config.COLLECTION_ANIMATION_WAIT)
                    
                except Exception as e:
                    logger.error(f"钓鱼过程中出错: {e}")
                    # 出错后等待更长时间再重试
                    time.sleep(Config.FISHING_ERROR_RETRY_DELAY)
                    
        except KeyboardInterrupt:
            logger.info("自动钓鱼任务已停止")
        except Exception as e:
            logger.critical(f"自动钓鱼任务异常终止: {e}")
            track_usage_statistics("auto_fish", success=False)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='咸鱼助手 - 游戏辅助工具')
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4], help='要执行的任务编号 (1: 自动推图, 2: 自动推塔, 3: 自动答题, 4: 自动钓鱼)')
    parser.add_argument('--duration', type=int, help='任务持续时间（秒），0表示无限时间')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='日志级别 (默认: INFO)')
    return parser.parse_args()

if __name__ == '__main__':
    """主函数入口"""
    logger.info("=== 咸鱼之王辅助工具启动 ===")
    
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置日志级别（通过loguru的方式设置）
        logger.remove()
        logger.add(sys.stderr, level=args.log_level)
        logger.info(f"日志级别已设置为: {args.log_level}")
        
        # 显示版本信息
        logger.info(f"咸鱼助手 v{Config.VERSION}")
        logger.debug(f"运行环境: Python {sys.version}")
        
        # 记录程序启动
        track_usage_statistics("program_start")
        
        # 加载版本历史并检查游戏版本
        version_history = load_version_history()
        
        # 这里可以添加从游戏中提取版本号的逻辑
        # 目前使用默认版本1.0.0进行示例
        current_game_version = "1.0.0"
        logger.debug(f"当前检测到的游戏版本: {current_game_version}")
        
        # 检查游戏版本
        if not check_game_version(current_game_version):
            logger.warning("游戏版本可能不兼容，部分功能可能无法正常工作")
        else:
            logger.debug("游戏版本兼容性检查通过")
        
        # 更新版本历史
        version_history["last_check"] = datetime.now().isoformat()
        version_history["versions"].append({
            "version": current_game_version,
            "check_time": datetime.now().isoformat()
        })
        # 只保留最近10条记录
        if len(version_history["versions"]) > 10:
            version_history["versions"] = version_history["versions"][-10:]
        save_version_history(version_history)
        
        # 初始化OCR（根据配置决定是否启用GPU）
        logger.info("初始化OCR引擎...")
        logger.debug(f"OCR配置 - GPU模式: {Config.OCR_GPU_ENABLED}")
        
        # easyocr.Reader的第一个参数是语言列表
        ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=Config.OCR_GPU_ENABLED)
        logger.info("OCR引擎初始化完成")
        
        # 初始化辅助工具
        logger.info("初始化游戏辅助...")
        helper = XianYuHelper(ocr_reader)
        logger.info("游戏辅助初始化完成")
        logger.debug(f"游戏窗口标题: {Config.GAME_WINDOW_TITLE}")
        
        # 保存任务持续时间参数
        task_duration = args.duration if args.duration != 0 else None  # 0表示无限时间
        
        if args.task is None:
            print("================================")
            print("===== 咸鱼之王辅助工具 =====")
            print("请选择要执行的任务：")
            print("1. 自动推图，优化版")
            print("2. 自动推塔，待优化")
            print("3. 自动答题，待优化")
            print("4. 自动钓鱼，待优化")
            print("5. 识别底部按钮（咸将、装备、战斗、宝箱、副本）")
            print("================================")
            print("请输入 1-5 之间的数字，然后按回车：")
            choice = input()
            logger.info(f"用户选择的任务: {choice}")
        else:
            choice = str(args.task)
            logger.info(f"通过命令行指定任务: {choice}")
        
        if choice == '1':
            logger.info("开始执行自动推图任务...")
            helper.task_auto_pass(duration=task_duration)
            track_usage_statistics("auto_pass", success=True)
        elif choice == '2':
            logger.info("开始执行自动推塔任务...")
            helper.task_auto_tower(duration=task_duration)
            track_usage_statistics("auto_tower", success=True)
        elif choice == '3':
            logger.info("开始执行自动答题任务...")
            helper.task_auto_answer(duration=task_duration)
            track_usage_statistics("auto_answer", success=True)
        elif choice == '4':
            logger.info("开始执行自动钓鱼任务...")
            helper.task_auto_fish(duration=task_duration)
            track_usage_statistics("auto_fish", success=True)
        elif choice == '5':
            logger.info("开始执行底部按钮识别任务...")
            print("正在识别底部按钮...")
            print("注意：调试截图已保存到 debug_screenshots 目录中")
            
            try:
                # 识别底部按钮
                buttons = helper.recognize_bottom_buttons()
                
                # 统计识别成功率
                recognized_count = sum(1 for btn in buttons if btn is not None)
                total_count = len(buttons) if buttons else 5  # 默认应该有5个按钮
                
                print(f"\n底部按钮识别结果（从左到右）：")
                print(f"识别成功率: {recognized_count}/{total_count}")
                print("----------------------------------------")
                
                # 定义预期的按钮顺序，用于更好的显示
                expected_buttons = ['咸将', '装备', '战斗', '宝箱', '副本']
                
                for i in range(5):  # 固定显示5个位置
                    if i < len(buttons) and buttons[i]:
                        print(f"{i+1}. {buttons[i]} ✓")
                    else:
                        # 如果没有识别到，显示预期的按钮名称作为提示
                        expected_name = expected_buttons[i] if i < len(expected_buttons) else "未知"
                        print(f"{i+1}. 未识别到（预期: {expected_name}）✗")
                
                print("----------------------------------------")
                
                # 如果识别到至少一个按钮，提供点击选项
                if recognized_count > 0:
                    # 询问用户是否要点击某个按钮
                    print("\n是否要点击某个底部按钮？(y/n)")
                    click_choice = input().strip().lower()
                    if click_choice == 'y':
                        print("请输入要点击的按钮名称（咸将、装备、战斗、宝箱、副本）：")
                        button_to_click = input().strip()
                        
                        print(f"正在尝试点击按钮: {button_to_click}...")
                        success = helper.click_bottom_button(button_to_click)
                        if success:
                            print(f"✓ 成功点击按钮：{button_to_click}")
                        else:
                            print(f"✗ 点击按钮失败：{button_to_click}")
                            print("提示：如果游戏窗口未被正确识别，程序会尝试使用备用坐标")
                            print("您可以检查 debug_screenshots 目录中的截图以分析问题")
                else:
                    print("\n未能识别到任何按钮，请检查：")
                    print("1. 游戏窗口是否可见")
                    print("2. debug_screenshots 目录中的截图是否正确捕获了底部按钮区域")
                    print("3. 可能需要调整 config.py 中的按钮区域坐标")
                    
            except Exception as e:
                logger.error(f"执行底部按钮识别任务时出错: {str(e)}")
                print(f"\n识别过程中发生错误: {str(e)}")
                print("请查看日志文件了解详细信息")
                print("建议检查游戏窗口是否正常显示")
            finally:
                track_usage_statistics("recognize_bottom_buttons", success=True)
        else:
            logger.error("无效的选择，请重新运行程序并输入1-5之间的数字")
            print("无效的选择，请重新运行程序并输入1-5之间的数字")
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        track_usage_statistics("program_exit", success=True)
    except Exception as e:
        logger.critical(f"程序异常终止: {str(e)}")
        track_usage_statistics("program_exit", success=False)
    finally:
        logger.info("开始清理资源...")
        # 使用上下文管理器的思想进行资源清理
        # 释放OCR引擎资源
        if 'ocr_reader' in locals():
            try:
                # EasyOCR的清理方法
                if hasattr(ocr_reader, 'close'):
                    logger.debug("释放OCR引擎资源...")
                    ocr_reader.close()
                    logger.info("OCR引擎资源已释放")
                # 对于不支持close方法的版本，尝试手动清理内存
                elif hasattr(ocr_reader, 'reader'):
                    logger.debug("尝试清理OCR引擎内部资源...")
                    ocr_reader.reader = None
            except Exception as e:
                logger.warning(f"释放OCR引擎资源时出错: {str(e)}")
        
        # 释放GameHelper可能占用的资源
        if 'helper' in locals():
            try:
                # 如果GameHelper有清理方法，调用它
                if hasattr(helper.game_helper, 'cleanup'):
                    logger.debug("清理GameHelper资源...")
                    helper.game_helper.cleanup()
                logger.info("游戏辅助资源已清理")
            except Exception as e:
                logger.warning(f"清理游戏辅助资源时出错: {str(e)}")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        logger.debug("执行垃圾回收")
        
        logger.info("资源清理完成")
        logger.info("=== 咸鱼之王辅助工具退出 ===")