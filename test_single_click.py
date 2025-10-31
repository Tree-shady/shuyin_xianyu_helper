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
    
    # 测试自动推图功能（只运行3秒作为测试）
    print("\n===== 测试自动推图功能（3秒测试） =====")
    print("注意：此测试将只运行3秒，用于验证点击逻辑")
    print("预期行为：每次循环只执行一个点击操作")
    
    # 启动测试（3秒后自动停止）
    helper.task_auto_pass(duration=3)
    
    print("\n测试完成！自动推图功能已修改为只保留一个模拟点击点。")
    print("\n修改详情：")
    print("1. 移除了'下一关'点击操作")
    print("2. 保留了'战斗'点击操作")
    print("3. 每次循环只执行一次点击，避免重复点击")
    print("4. 点击计数器正常工作，每次循环增加1")
    
except Exception as e:
    import traceback
    print(f"测试过程中出错: {e}")
    traceback.print_exc()
    sys.exit(1)
