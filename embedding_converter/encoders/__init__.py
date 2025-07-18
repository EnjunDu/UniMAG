# -*- coding: utf-8 -*-
"""
编码器模块自动发现与注册

这个 __init__.py 文件实现了动态导入机制，它会自动扫描当前目录下的所有
Python 文件，并导入它们。这确保了所有使用 @EncoderFactory.register 装饰器
的编码器类都能被自动注册到工厂中，无需在主程序中手动导入每一个编码器文件。

实现真正的“即插即用”：
1. 将新的编码器实现（如 my_new_encoder.py）放入此 encoders/ 目录。
2. 确保新编码器类使用了 @EncoderFactory.register("my_new_encoder") 装饰器。
3. 系统将自动发现并使其可用，无需修改任何其他代码。
"""
import os
import importlib
from pathlib import Path

# 获取当前文件所在的目录
package_dir = Path(__file__).resolve().parent

# 遍历目录中的所有文件
for filename in os.listdir(package_dir):
    # 检查文件是否为Python文件，且不是__init__.py本身
    if filename.endswith(".py") and filename != "__init__.py":
        # 从文件名构造模块名 (例如, 'qwen_vl_encoder.py' -> 'qwen_vl_encoder')
        module_name = filename[:-3]
        
        # 动态导入模块
        # 使用相对导入，格式为 .<module_name>
        # 例如, importlib.import_module('.qwen_vl_encoder', package='embedding_converter.encoders')
        importlib.import_module(f".{module_name}", package=__name__)