#!/usr/bin/env python3
"""
使用新数据格式运行OpenGraph的示例脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import *

def run_with_new_dataset():
    """使用新数据集格式运行OpenGraph"""
    
    # 设置参数
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if len(args.gpu.split(',')) > 1:
        args.devices = ['cuda:0', 'cuda:1']
    else:
        args.devices = ['cuda:0', 'cuda:0']
    args.devices = list(map(lambda x: t.device(x), args.devices))
    logger.saveDefault = True
    setproctitle.setproctitle('OpenGraph-NewFormat')

    log('Starting OpenGraph with new dataset format')
    
    # 使用新数据集格式
    # 这里可以根据需要修改数据集名称
    if hasattr(args, 'trndata') and args.trndata:
        trn_datasets = [args.trndata]
    else:
        trn_datasets = ['cloth-copurchase']  # 默认使用cloth-copurchase
        
    if hasattr(args, 'tstdata') and args.tstdata:
        tst_datasets = [args.tstdata]
    else:
        tst_datasets = ['cloth-copurchase']  # 默认使用cloth-copurchase

    trn_datasets = list(set(trn_datasets))
    tst_datasets = list(set(tst_datasets))
    
    log(f'Training datasets: {trn_datasets}')
    log(f'Testing datasets: {tst_datasets}')
    
    # 创建多数据处理器
    multi_handler = MultiDataHandler(trn_datasets, tst_datasets)
    log('Data loaded successfully')

    # 运行实验
    exp = Exp(multi_handler)
    exp.run()

if __name__ == '__main__':
    run_with_new_dataset()
