import unittest
import sys
import os
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xianyu import XianYuHelper
from config import Config

class TestOCR(unittest.TestCase):
    """测试OCR识别功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟的OCR阅读器
        self.mock_ocr_reader = Mock()
        # 创建XianYuHelper实例，使用模拟的OCR阅读器
        self.helper = XianYuHelper(self.mock_ocr_reader)
    
    def test_recognize_level_successful(self):
        """测试成功识别关卡"""
        # 设置模拟的OCR识别结果
        self.mock_ocr_reader.readtext.return_value = [[[[0,0],[10,0],[10,10],[0,10]], '12', 0.95]]
        
        # 模拟grab_screen_region方法
        with patch('utils.GameHelper.grab_screen_region') as mock_grab:
            # 模拟返回一个有效的图像
            mock_grab.return_value = Mock()
            
            # 调用_recognize_level方法（需要访问私有方法，这里仅作为示例）
            # 注意：在实际测试中，可能需要测试使用该方法的公共接口
            # 由于是私有方法，这里使用反射来访问
            if hasattr(self.helper, '_recognize_level'):
                level_region = Config.LEVEL_RECOGNITION_REGION
                result = self.helper._recognize_level(level_region)
                self.assertEqual(result, '12')
    
    def test_recognize_level_empty_result(self):
        """测试OCR返回空结果的情况"""
        # 设置模拟的OCR识别结果为空
        self.mock_ocr_reader.readtext.return_value = []
        
        with patch('utils.GameHelper.grab_screen_region') as mock_grab:
            mock_grab.return_value = Mock()
            
            if hasattr(self.helper, '_recognize_level'):
                level_region = Config.LEVEL_RECOGNITION_REGION
                result = self.helper._recognize_level(level_region)
                self.assertEqual(result, '')
    
    def test_recognize_level_error_handling(self):
        """测试OCR识别过程中的异常处理"""
        # 模拟OCR识别抛出异常
        self.mock_ocr_reader.readtext.side_effect = Exception("OCR error")
        
        with patch('utils.GameHelper.grab_screen_region') as mock_grab:
            mock_grab.return_value = Mock()
            
            # 测试应该不会崩溃，而是返回空字符串
            if hasattr(self.helper, '_recognize_level'):
                level_region = Config.LEVEL_RECOGNITION_REGION
                try:
                    result = self.helper._recognize_level(level_region)
                    # 假设异常被捕获并返回空字符串
                    self.assertEqual(result, '')
                except Exception:
                    self.fail("_recognize_level方法在OCR错误时应该捕获异常")
    
    @patch('logging.error')
    def test_game_helper_ocr_integration(self, mock_logging_error):
        """测试GameHelper与OCR的集成"""
        # 验证GameHelper是否正确使用了OCR阅读器
        self.assertEqual(self.helper.game_helper.reader, self.mock_ocr_reader)
        
        # 模拟grab_screen_region和OCR识别
        with patch('utils.GameHelper.grab_screen_region') as mock_grab:
            mock_grab.return_value = Mock()
            self.mock_ocr_reader.readtext.return_value = [[[[0,0],[10,0],[10,10],[0,10]], '测试文本', 0.9]]
            
            # 这里可以添加更多测试，具体取决于GameHelper类中使用OCR的方法

if __name__ == '__main__':
    unittest.main()