#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""关卡识别集成测试脚本"""

import os
import sys
from xianyu import XianYuHelper
from level_recognition import LevelRecognizer
from config import Config

def test_integration():
    """测试关卡识别功能集成"""
    print("===== 关卡识别集成测试 =====")
    
    # 测试LevelRecognizer类
    print("\n1. 测试LevelRecognizer类")
    try:
        recognizer = LevelRecognizer()
        print("   ✓ LevelRecognizer实例化成功")
        print(f"   ✓ 识别器属性检查: hwnd={recognizer.hwnd}")
        
        # 尝试查找游戏窗口
        found = recognizer.find_game_window()
        print(f"   ✓ 游戏窗口查找: {'成功' if found else '未找到'}")
    except Exception as e:
        print(f"   ✗ LevelRecognizer测试失败: {e}")
    
    # 测试XianYuHelper集成
    print("\n2. 测试XianYuHelper集成")
    try:
        # 先初始化OCR引擎
        import easyocr
        print("   ✓ 初始化OCR引擎")
        ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=Config.OCR_GPU_ENABLED)
        
        helper = XianYuHelper(ocr_reader)
        print("   ✓ XianYuHelper实例化成功")
        
        if hasattr(helper, 'level_recognizer'):
            print("   ✓ 关卡识别器属性存在")
            if helper.level_recognizer:
                print("   ✓ 关卡识别器初始化成功")
            else:
                print("   ⚠ 关卡识别器初始化失败但属性存在")
        else:
            print("   ✗ 关卡识别器属性不存在")
        
        # 检查配置
        print("\n3. 配置检查")
        print(f"   ✓ 关卡识别区域配置: {Config.LEVEL_RECOGNITION_REGION}")
        
        # 检查推图功能配置
        print(f"   ✓ 推图点击位置: {Config.PASS_LEVEL_CLICKS}")
        
    except Exception as e:
        print(f"   ✗ XianYuHelper测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n===== 测试完成 =====")

if __name__ == "__main__":
    test_integration()
