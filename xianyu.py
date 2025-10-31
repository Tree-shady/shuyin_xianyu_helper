import json
import re
import sys
import time
import warnings
import easyocr
import win32gui
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
                    # 执行点击操作推进关卡（只保留一个点击点）
                    logger.debug(f"循环 {loop_count}: 执行战斗点击 - 坐标({Config.PASS_LEVEL_CLICKS['battle'][0]}, {Config.PASS_LEVEL_CLICKS['battle'][1]})")
                    self.game_helper.left_click(Config.PASS_LEVEL_CLICKS['battle'][0], 
                                              Config.PASS_LEVEL_CLICKS['battle'][1], delay=0.01)
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
            print("================================")
            print("请输入 1-4 之间的数字，然后按回车：")
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
        else:
            logger.error("无效的选择，请重新运行程序并输入1-4之间的数字")
            print("无效的选择，请重新运行程序并输入1-4之间的数字")
        
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