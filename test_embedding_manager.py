import sys
from pathlib import Path

# 确保 'utils' 目录在Python的搜索路径中
# 这样我们就可以从项目根目录运行此脚本
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager

def run_tests():
    """
    运行一系列测试来验证 EmbeddingManager 的功能。
    """
    print("=== 开始测试 EmbeddingManager ===")
    
    # 1. 初始化管理器，并指定本地数据集的根目录
    # 这个路径是相对于你运行此脚本的位置
    local_data_path = "/home/ai/ylzuo/UniMAG/hugging_face"
    manager = EmbeddingManager(base_path=local_data_path)
    print(f"使用数据集根目录: '{Path(local_data_path).resolve()}'\n")

    # 2. 定义要测试的参数组合
    dataset_to_test = "books-nc-50"
    encoder_to_test = "Qwen/Qwen2.5-VL-3B-Instruct"

    tests = [
        {
            "description": "测试文本特征 (768维)",
            "params": {
                "dataset_name": dataset_to_test,
                "modality": "text",
                "encoder_name": encoder_to_test,
                "dimension": 768,
            },
        },
        {
            "description": "测试图像特征 (原生维度)",
            "params": {
                "dataset_name": dataset_to_test,
                "modality": "image",
                "encoder_name": encoder_to_test,
                "dimension": None, # 测试获取原生维度
            },
        },
        {
            "description": "测试多模态特征 (原生维度)",
            "params": {
                "dataset_name": dataset_to_test,
                "modality": "multimodal",
                "encoder_name": encoder_to_test,
                "dimension": None,
            },
        },
        {
            "description": "测试一个不存在的维度 (预期失败)",
            "params": {
                "dataset_name": dataset_to_test,
                "modality": "text",
                "encoder_name": encoder_to_test,
                "dimension": 9999, # 一个假设不存在的维度
            },
        },
    ]

    # 3. 循环执行测试
    all_tests_passed = True
    for i, test in enumerate(tests):
        print(f"--- [测试 {i+1}/{len(tests)}] {test['description']} ---")
        
        embedding = manager.get_embedding(**test["params"])
        
        if embedding is not None:
            print(f"  [成功] 获取到特征向量，形状: {embedding.shape}, 数据类型: {embedding.dtype}")
            
            # 如果获取成功，则进一步调用 view_embedding 来预览内容
            print("  [预览] 调用 view_embedding 查看向量内容...")
            manager.view_embedding(**test["params"])

        else:
            # 对于预期失败的测试，这是一个成功的结果
            if "预期失败" in test["description"]:
                print("  [成功] 如预期一样，未找到对应的嵌入文件。")
            else:
                print("  [失败] 未能获取到特征向量。请检查路径和文件名是否正确。")
                all_tests_passed = False
        print("-" * (len(test["description"]) + 20))
        print()

    # 4. 总结测试结果
    print("="*20)
    if all_tests_passed:
        print("✅ 所有预期成功的测试均已通过！")
    else:
        print("❌ 部分测试失败，请检查上面的日志。")
    print("="*20)


if __name__ == "__main__":
    run_tests()