import re
import os
import cv2
import time
import difflib
import windows
import win32gui
import easyocr
import numpy as np
import json
from loguru import logger
from PIL import ImageGrab, Image, ImageDraw, ImageFont
from collections import OrderedDict



def replace_chinese(file):
    """去除字符串中的中文"""
    pattern = re.compile(r'[^一-龥]')
    chinese = re.sub(pattern, '', file)
    return chinese


def stri_similar(s1, s2):
    """比较两个字符串的相似度"""
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def retry_operation(operation, max_retries=3, delay=0.5, *args, **kwargs):
    """
    重试操作的通用函数
    
    Args:
        operation: 要执行的函数
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
        *args: 传递给operation的位置参数
        **kwargs: 传递给operation的关键字参数
    
    Returns:
        operation的返回值
    
    Raises:
        如果所有重试都失败，则抛出最后一次的异常
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"操作失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    logger.error(f"所有重试都失败: {last_exception}")
    raise last_exception


def check_game_version(current_version, expected_version="1.0.0"):
    """
    检查游戏版本是否匹配
    
    Args:
        current_version: 当前检测到的游戏版本
        expected_version: 预期的游戏版本
    
    Returns:
        bool: 如果版本匹配或兼容，返回True；否则返回False
    """
    try:
        # 简单的版本比较逻辑
        current_parts = [int(part) for part in current_version.split(".")]
        expected_parts = [int(part) for part in expected_version.split(".")]
        
        # 主版本号必须匹配
        if current_parts[0] != expected_parts[0]:
            logger.warning(f"游戏主版本不匹配: 检测到 {current_version}, 期望 {expected_version}")
            return False
        
        # 次版本号如果更高，认为是兼容的
        if len(current_parts) > 1 and len(expected_parts) > 1:
            if current_parts[1] < expected_parts[1]:
                logger.warning(f"游戏次版本低于期望: 检测到 {current_version}, 期望 {expected_version}")
                return False
        
        logger.info(f"游戏版本检查通过: {current_version}")
        return True
    except Exception as e:
        logger.error(f"版本检查失败: {e}")
        return False


def load_version_history():
    """
    加载版本历史记录
    
    Returns:
        dict: 版本历史记录
    """
    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "version_history.json")
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"加载版本历史失败: {e}")
    return {"last_check": None, "versions": []}


def save_version_history(history):
    """
    保存版本历史记录
    
    Args:
        history: 版本历史记录字典
    """
    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "version_history.json")
    try:
        # 确保data目录存在
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存版本历史失败: {e}")


def track_usage_statistics(feature_name, success=True):
    """
    记录使用统计数据
    
    Args:
        feature_name: 功能名称
        success: 是否成功执行
    """
    stats_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "usage_stats.json")
    try:
        # 加载现有统计数据
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        else:
            stats = {"total_runs": 0, "features": {}}
        
        # 更新统计数据
        stats["total_runs"] += 1
        
        if feature_name not in stats["features"]:
            stats["features"][feature_name] = {"count": 0, "success": 0, "failure": 0}
        
        stats["features"][feature_name]["count"] += 1
        if success:
            stats["features"][feature_name]["success"] += 1
        else:
            stats["features"][feature_name]["failure"] += 1
        
        # 保存更新后的统计数据
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    except Exception as e:
        # 统计失败不应影响主程序
        logger.debug(f"记录使用统计失败: {e}")


class GameHelper:
    """
    游戏辅助的通用工具类
    负责窗口管理、OCR识别、图像处理等通用功能
    """
    def __init__(self, ocr_reader, window_class="Chrome_WidgetWin_0", window_name="咸鱼之王"):
        """
        初始化游戏辅助工具
        
        Args:
            ocr_reader: EasyOCR阅读器实例
            window_class: 游戏窗口类名
            window_name: 游戏窗口标题
        """
        # 窗口信息
        self.window_class = window_class
        self.window_name = window_name
        
        # 窗口句柄和坐标
        self.hwnd = None
        self.left, self.top, self.right, self.bottom = 0, 0, 0, 0
        
        # OCR相关
        self.reader = ocr_reader
        
        # 改进的OCR缓存系统 - 使用LRU策略
        self.ocr_cache = OrderedDict()  # 有序字典，支持LRU
        self.cache_max_size = 100  # 最大缓存条目数
        self.cache_hit_count = 0  # 缓存命中计数
        self.cache_miss_count = 0  # 缓存未命中计数
        
        # 调试设置
        self.debug = True
        self.capture_method = "foreground"  # 前台窗口截图
        
        # 分辨率相关
        self.screen_height = 2560
        self.screen_width = 1440
        self.scale_percentage = 100
        
        # 图像匹配阈值
        self.threshold = 0.90
        
        # 初始化窗口
        self.initialize_window()
        
        logger.info("游戏辅助工具初始化完成")
    
    def initialize_window(self):
        """初始化窗口句柄和坐标"""
        self.hwnd = retry_operation(
            windows.find_hwd, 
            max_retries=3, 
            delay=1,
            class_name=self.window_class, 
            window_name=self.window_name
        )
        
        if not self.hwnd:
            raise RuntimeError(f"找不到窗口: {self.window_name} (类: {self.window_class})")
        
        self.left, self.top, self.right, self.bottom = retry_operation(
            windows.get_window_pos, 
            max_retries=3, 
            delay=0.5,
            hwnd=self.hwnd
        )
        
        logger.info("窗口坐标：{} {} {} {}", self.left, self.top, self.right, self.bottom)
    
    def check_window(self):
        """检查窗口是否存在且可见，不存在则尝试重新获取"""
        if not self.hwnd or not win32gui.IsWindow(self.hwnd) or not win32gui.IsWindowVisible(self.hwnd):
            logger.warning("窗口不可见或已关闭，尝试重新获取...")
            try:
                self.initialize_window()
                return True
            except Exception as e:
                logger.error(f"重新获取窗口失败: {e}")
                return False
        return True
    
    def set_foreground(self):
        """设置窗口为前台"""
        try:
            if self.check_window():
                win32gui.SetForegroundWindow(self.hwnd)
                return True
            return False
        except Exception as e:
            logger.error(f"设置窗口为前台失败: {e}")
            return False
            
    def cleanup(self):
        """
        清理资源
        用于程序结束时释放占用的资源
        """
        try:
            # 清空缓存，释放内存
            if hasattr(self, 'ocr_cache'):
                self.ocr_cache.clear()
                logger.debug("OCR缓存已清空")
            
            # 释放窗口相关资源
            # 注意：在Python中，win32gui资源通常由垃圾回收处理
            # 这里主要是设置为None，方便垃圾回收
            self.hwnd = None
            
            logger.debug("GameHelper资源已清理")
        except Exception as e:
            logger.error(f"清理GameHelper资源时出错: {e}")
    
    def get_ocr_result(self, img, use_cache=True):
        """
        获取OCR识别结果，使用LRU缓存策略提高性能
        
        Args:
            img: 要识别的图像
            use_cache: 是否使用缓存
            
        Returns:
            OCR识别结果列表
        """
        # 定期清理缓存，避免内存占用过大
        if len(self.ocr_cache) > self.cache_max_size * 1.5:
            self._clean_cache()
            
        # 记录统计信息（每100次调用）
        if (self.cache_hit_count + self.cache_miss_count) % 100 == 0 and (self.cache_hit_count + self.cache_miss_count) > 0:
            hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100
            logger.debug(f"OCR缓存统计: 命中={self.cache_hit_count}, 未命中={self.cache_miss_count}, 命中率={hit_rate:.2f}%")
        
        if use_cache and isinstance(img, np.ndarray):
            # 使用图像哈希作为缓存键
            img_hash = hash(img.tobytes())
            
            # 检查缓存是否命中
            if img_hash in self.ocr_cache:
                # 缓存命中，将条目移到末尾（表示最近使用）
                result = self.ocr_cache[img_hash]
                # 移除后再添加到末尾
                self.ocr_cache.pop(img_hash)
                self.ocr_cache[img_hash] = result
                self.cache_hit_count += 1
                return result
            
            # 缓存未命中，执行OCR
            self.cache_miss_count += 1
            result = self.reader.readtext(img)
            
            # 添加到缓存
            self._add_to_cache(img_hash, result)
            return result
        else:
            # 不使用缓存或图像类型不支持缓存
            return self.reader.readtext(img)
    
    def _add_to_cache(self, key, value):
        """
        添加项目到缓存，并维护LRU顺序
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        # 如果缓存已满，移除最少使用的条目（OrderedDict的第一个）
        if len(self.ocr_cache) >= self.cache_max_size:
            self.ocr_cache.popitem(last=False)
        
        # 添加新条目到末尾（最近使用）
        self.ocr_cache[key] = value
    
    def _clean_cache(self):
        """
        清理缓存，减少内存占用
        """
        # 保留最近使用的一半缓存
        keep_size = max(10, int(self.cache_max_size / 2))
        
        # 移除最早的条目
        while len(self.ocr_cache) > keep_size:
            self.ocr_cache.popitem(last=False)
        
        logger.debug(f"清理OCR缓存，保留 {keep_size} 个最近使用的条目")
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计的字典
        """
        total = self.cache_hit_count + self.cache_miss_count
        hit_rate = self.cache_hit_count / total * 100 if total > 0 else 0
        
        return {
            "size": len(self.ocr_cache),
            "max_size": self.cache_max_size,
            "hits": self.cache_hit_count,
            "misses": self.cache_miss_count,
            "total": total,
            "hit_rate": hit_rate
        }
    
    def preprocess_image(self, img, color_thresholds=None):
        """
        预处理图像以提高OCR识别准确率
        
        Args:
            img: PIL Image对象
            color_thresholds: 颜色阈值列表，格式为 [[r, g, b, tolerance], ...]
            
        Returns:
            处理后的图像
        """
        if color_thresholds is None:
            # 默认颜色阈值
            color_thresholds = [[48, 48, 48, 32], [82, 31, 26, 40]]
        
        # 转换为numpy数组进行批量处理
        img_np = np.array(img)
        
        # 创建掩码
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        for color in color_thresholds:
            lower = np.array([color[0]-color[3], color[1]-color[3], color[2]-color[3]])
            upper = np.array([color[0]+color[3], color[1]+color[3], color[2]+color[3]])
            color_mask = cv2.inRange(img_np, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        
        # 应用掩码
        img_np[mask > 0] = [0, 0, 0]  # 黑色
        img_np[mask == 0] = [255, 255, 255]  # 白色
        
        return Image.fromarray(img_np)
    
    def grab_screen_region(self, region=None):
        """
        截取屏幕特定区域
        
        Args:
            region: 区域坐标 (left, top, right, bottom)，None表示整个窗口
            
        Returns:
            截取的图像
        """
        if not self.check_window():
            raise RuntimeError("无法截取屏幕：窗口不存在或不可见")
        
        if region is None:
            # 截取整个窗口
            bbox = (self.left, self.top, self.right, self.bottom)
        else:
            # 截取指定区域
            bbox = region
        
        try:
            return ImageGrab.grab(bbox=bbox)
        except Exception as e:
            logger.error(f"截取屏幕失败: {e}")
            raise
    
    def find_text_in_image(self, img, keyword, threshold=0.6):
        """
        在图像中查找文本
        
        Args:
            img: 要搜索的图像
            keyword: 要查找的关键字
            threshold: 相似度阈值
            
        Returns:
            (是否找到, 最佳匹配文本, 相似度)
        """
        try:
            img_np = np.array(img)
            result = self.get_ocr_result(img_np)
            
            best_match = ""
            best_similarity = 0
            
            for item in result:
                text = item[1].replace(" ", "")  # 去除空格
                similarity = stri_similar(text, keyword)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = text
            
            return best_similarity > threshold, best_match, best_similarity
        except Exception as e:
            logger.error(f"查找文本失败: {e}")
            return False, "", 0
    
    def left_click(self, x, y, delay=0.01, max_retries=3):
        """
        点击指定位置，带重试机制
        
        Args:
            x: 相对窗口的X坐标
            y: 相对窗口的Y坐标
            delay: 点击后的延迟（秒）
            max_retries: 最大重试次数
        
        Returns:
            bool: 点击是否成功
        """
        # 自定义重试逻辑，支持基于返回值False的重试
        attempt = 0
        while attempt < max_retries:
            try:
                # 调用windows模块的left_click_position函数
                success = windows.left_click_position(
                    self.hwnd,
                    x_position=x,
                    y_position=y,
                    sleep_time=delay
                )
                
                if success:
                    logger.debug(f"点击成功 (尝试 {attempt+1}/{max_retries})")
                    return True
                else:
                    logger.warning(f"点击失败 (尝试 {attempt+1}/{max_retries})")
                    attempt += 1
                    if attempt < max_retries:
                        logger.debug(f"等待 {0.5} 秒后重试")
                        time.sleep(0.5)
            except Exception as e:
                logger.warning(f"点击操作抛出异常 (尝试 {attempt+1}/{max_retries}): {e}")
                attempt += 1
                if attempt < max_retries:
                    logger.debug(f"等待 {0.5} 秒后重试")
                    time.sleep(0.5)
        
        logger.error(f"所有 {max_retries} 次点击尝试都失败")
        return False

    # 加载图像资源, 返回图像资源列表; 图像都是通过Sniff截图，保持图片画质稳定
    def load_res(self):
        # 匹配对象的字典
        self.res = {}
        file_dir = os.path.join(os.getcwd(), "data")
        temp_list = os.listdir(file_dir)
        for item in temp_list:
            if item.endswith(".bmp"):
                self.res[item] = {}
                res_path = os.path.join(file_dir, item)
                # 路径包含中文无法使用cv2.imread读取
                # self.res[item]["img"] = cv2.imread(res_path)
                self.res[item]["img"] = cv2.imdecode(np.fromfile(res_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                # 图片读取失败
                if self.res[item]["img"] is None:
                    logger.warning(f"图片{item}读取失败")
                    continue
                # 如果不是原尺寸（1440P），进行对应缩放操作
                if self.scale_percentage != 100:
                    self.res[item]["width"] = int(self.res[item]["img"].shape[1] * self.scale_percentage / 100) 
                    self.res[item]["height"] = int(self.res[item]["img"].shape[0] * self.scale_percentage / 100)
                    self.res[item]["img"] = cv2.resize(self.res[item]["img"], (self.res[item]["width"], self.res[item]["height"]), interpolation=cv2.INTER_AREA)
                else:
                    self.res[item]["height"], self.res[item]["width"], self.res[item]["channel"] = self.res[item]["img"].shape[::]

                
    # 获取截图
    def get_img(self, pop_up_window=False, save_img=False, file_name='screenshot.png'):
        if self.capture_method == "foreground":
            # 前台窗口截图
            image_bytes = ImageGrab.grab(bbox=(self.left, self.top, self.right, self.bottom))
        else:
            image_bytes = windows.capture(self.hwnd)

        image_bytes = np.array(image_bytes, dtype=np.uint8)
        if image_bytes.size == 0:
            logger.warning("截图失败，检查窗口是否被遮挡")
        else:
            self.target_img = image_bytes
            if save_img:
                cv2.imwrite(file_name, self.target_img)
            if pop_up_window:
                self.show_img()

                

    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "data/simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


    # 展示图片
    def show_img(self):
        img = self.target_img        
        if self.debug:
            # 统计耗时
            start = time.time()
            result = self.get_ocr_result(img)  # 使用缓存版本
            end = time.time()
            
            # 获取缓存统计信息
            cache_stats = self.get_cache_stats()
            logger.debug("OCR耗时：{}ms, 缓存命中率：{:.2f}%", 
                       (end - start) * 1000, 
                       cache_stats['hit_rate'])
            spacer = 100
            for detection in result:
                # 异常保护
                try:
                    top_left = tuple(detection[0][0])
                    bottom_right = tuple(detection[0][2])
                    text = detection[1]
                    img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),1)
                    #img = self.cv2ImgAddText(img, text, 100, 100 + spacer)
                    img = self.cv2ImgAddText(img, text, top_left[0], top_left[1] + 20)
                    spacer+=20
                except Exception as e:
                    logger.error(e)
                    continue          
        
        #cv2.namedWindow("screenshot", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('screenshot', 360, 640)
        cv2.imshow("screenshot", img)
        cv2.waitKey()


if __name__ == '__main__':
    # 测试代码
    try:
        helper = GameHelper(easyocr.Reader(['ch_sim', 'en'], gpu=False))
        helper.set_foreground()
        img = helper.grab_screen_region()
        logger.info(f"成功截取图像: {img.size}")
    except Exception as e:
        logger.error(f"测试失败: {e}")
