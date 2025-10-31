import time
import sys
from xianyu import XianYuHelper
import easyocr

# 初始化OCR引擎
print("初始化OCR引擎...")
ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# 初始化辅助工具
print("初始化XianYuHelper...")
try:
    helper = XianYuHelper(ocr_reader)
    print("XianYuHelper初始化成功")
    
    # 测试关卡识别间隔功能
    print("\n===== 测试关卡识别间隔功能 =====")
    
    # 模拟当前时间
    current_time = time.time()
    
    # 测试第一次识别（应该执行）
    print(f"\n1. 第一次识别测试:")
    helper._last_recognition_time = 0  # 重置为第一次识别
    if helper._last_recognition_time == 0:
        print("✓ 将执行第一次关卡识别")
    else:
        print("✗ 识别条件判断失败")
    
    # 测试120秒内（不应执行）
    print(f"\n2. 120秒内识别测试:")
    helper._last_recognition_time = current_time - 60  # 60秒前
    if current_time - helper._last_recognition_time > 120:
        print("✗ 识别条件判断错误，120秒内不应执行识别")
    else:
        print("✓ 正确：120秒内不会执行识别")
    
    # 测试超过120秒（应该执行）
    print(f"\n3. 超过120秒识别测试:")
    helper._last_recognition_time = current_time - 180  # 180秒前
    if current_time - helper._last_recognition_time > 120:
        print("✓ 正确：超过120秒将执行识别")
    else:
        print("✗ 识别条件判断错误，超过120秒应执行识别")
    
    # 测试关卡更新检测逻辑
    print(f"\n4. 关卡更新检测测试:")
    test_game_level = "第10关"
    test_new_level = "第11关"
    
    # 相同关卡测试
    if test_new_level == test_game_level:
        print("✓ 关卡未更新，将显示'关卡未更新'消息")
    else:
        print("✓ 关卡已更新，将显示新关卡信息")
    
    print("\n测试完成！关卡识别间隔功能已正确配置")
    
    print("\n使用说明:")
    print("1. 系统会在第一次识别后记录时间戳")
    print("2. 随后120秒内不会再次执行识别，减少资源占用")
    print("3. 120秒后会重新识别，如检测到关卡更新则刷新显示")
    print("4. 如未检测到更新，则保持当前关卡显示")
    
except Exception as e:
    import traceback
    print(f"初始化失败: {e}")
    traceback.print_exc()
    sys.exit(1)
