import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    从YAML文件加载和管理配置
    """

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从指定的路径加载YAML配置文件。

        Args:
            config_path: 配置文件的路径。

        Returns:
            一个包含配置内容的字典。
            
        Raises:
            FileNotFoundError: 如果配置文件不存在。
            yaml.YAMLError: 如果文件格式不正确。
        """
        config_path = Path(config_path)
        if not config_path.is_file():
            logger.error(f"配置文件未找到: {config_path}")
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        logger.info(f"正在从 {config_path} 加载配置...")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("配置加载成功。")
            return config
        except yaml.YAMLError as e:
            logger.error(f"解析YAML文件时出错: {config_path}\n{e}")
            raise

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        验证配置字典是否包含必要的顶级键。

        Args:
            config: 配置字典。

        Returns:
            如果配置有效则返回True，否则返回False。
        """
        required_keys = [
            "pipeline_settings",
            "encoder_settings",
            "dataset_settings"
        ]
        
        for key in required_keys:
            if key not in config:
                logger.error(f"配置中缺少必需的键: '{key}'")
                return False
        
        logger.info("配置验证通过。")
        return True

# 便捷函数
def load_embedding_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载并验证嵌入管道的配置文件。

    Args:
        config_path: 配置文件的路径。

    Returns:
        经过验证的配置字典。
        
    Raises:
        ValueError: 如果配置无效。
    """
    config = ConfigLoader.load_config(config_path)
    if not ConfigLoader.validate_config(config):
        raise ValueError("配置文件无效，请检查日志获取详细信息。")
    return config

if __name__ == '__main__':
    # 测试配置加载器
    # 假设项目根目录下有 configs/embedding/qwen2.5-vl-3b.yaml
    try:
        default_config_path = Path(__file__).parent.parent / "configs" / "embedding" / "qwen2.5-vl-3b.yaml"
        config_data = load_embedding_config(default_config_path)
        
        print("配置加载和验证成功！")
        import json
        print(json.dumps(config_data, indent=2, ensure_ascii=False))
        
    except (FileNotFoundError, ValueError) as e:
        print(f"测试失败: {e}")