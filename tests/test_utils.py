import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import stri_similar

class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    def test_stri_similar_exact_match(self):
        """测试完全匹配的字符串相似度"""
        s1 = "这是一个测试字符串"
        s2 = "这是一个测试字符串"
        similarity = stri_similar(s1, s2)
        self.assertAlmostEqual(similarity, 1.0, places=2)
    
    def test_stri_similar_partial_match(self):
        """测试部分匹配的字符串相似度"""
        s1 = "这是一个测试字符串"
        s2 = "这是另一个测试字符串"
        similarity = stri_similar(s1, s2)
        # 应该有较高的相似度，但小于1.0
        self.assertGreater(similarity, 0.7)
        self.assertLess(similarity, 1.0)
    
    def test_stri_similar_different_strings(self):
        """测试完全不同的字符串相似度"""
        s1 = "这是第一个字符串"
        s2 = "completely different string"
        similarity = stri_similar(s1, s2)
        # 应该有较低的相似度
        self.assertLess(similarity, 0.3)
    
    def test_stri_similar_empty_string(self):
        """测试空字符串的处理"""
        s1 = ""
        s2 = "测试字符串"
        similarity = stri_similar(s1, s2)
        self.assertEqual(similarity, 0.0)
        
        # 两个空字符串应该完全匹配
        similarity = stri_similar("", "")
        self.assertEqual(similarity, 1.0)
    
    def test_stri_similar_substring(self):
        """测试子字符串的相似度"""
        s1 = "测试"
        s2 = "这是一个测试字符串"
        similarity = stri_similar(s1, s2)
        # 应该有一定的相似度，但不是很高
        self.assertGreater(similarity, 0.1)
        self.assertLess(similarity, 0.5)

if __name__ == '__main__':
    unittest.main()