#!/bin/bash

echo "安装OpenGraph节点分类所需的依赖包..."
echo "======================================"

# 检查Python环境
echo "检查Python环境..."
python --version

# 安装基础依赖
echo "安装基础依赖..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scipy numpy scikit-learn

# 安装PyTorch Geometric (可选，如果不需要可以注释掉)
echo "安装PyTorch Geometric..."
pip install torch-geometric

# 或者如果不需要torch_geometric，可以修改代码移除这个依赖
echo "如果不需要torch_geometric，可以运行以下命令修改代码："
echo "sed -i 's/import torch_geometric.transforms as T/# import torch_geometric.transforms as T/' data_handler.py"

# 检查安装结果
echo "检查安装结果..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import scipy; print(f'SciPy版本: {scipy.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"

echo "依赖安装完成！"
