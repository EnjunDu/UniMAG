# flake8: noqa
# This file is used to ensure that all encoder modules are imported,
# which allows the @EncoderFactory.register decorator to work correctly.

import importlib
import pkgutil

# 自动发现并导入此目录下的所有模块
for _, module_name, _ in pkgutil.iter_modules(__path__):
    if not module_name.startswith('_'):
        importlib.import_module(f".{module_name}", __package__)