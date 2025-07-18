## Graph-to-Image (G2I) 脚本使用说明

本文档介绍如何使用项目根目录下的 `Graph2Image.py` 脚本，将已有的图像模态嵌入与文本模态嵌入（`.npy` 文件）输入到阿里云通义千问（Tongyi Qianwen）文生图 API，生成对应节点的图像。

---

### 一、前提条件

1. **Python 环境**：Python 3.8 及以上，建议使用虚拟环境（如 `venv`）。
2. **依赖包**：激活虚拟环境后执行：

   ```bash
   pip install numpy requests
   ```
3. **嵌入文件准备**：

   * 图像模态嵌入示例路径：
     `/root/UniMAG/data/Grocery/image_features/node0_image.npy`
   * 文本模态嵌入示例路径：
     `/root/UniMAG/data/Grocery/text_features/node0_text.npy`
4. **通义千问 API Key**：在终端设置环境变量：

   ```bash
   export QWEN_API_KEY="<你的通义千问密钥>"
   ```

---

### 二、脚本位置

```text
/root/UniMAG/Graph2Image.py
```

---

### 三、参数说明

| 参数                  | 类型    | 必选 | 默认值             | 说明                                |
| ------------------- | ----- | -- | --------------- | --------------------------------- |
| `--img-emb`         | `str` | 是  | —               | 图像模态嵌入文件路径（`.npy`）                |
| `--txt-emb`         | `str` | 是  | —               | 文本模态嵌入文件路径（`.npy`）                |
| `--top-k`           | `int` | 否  | `10`            | 在提示中展示前 K 维特征                     |
| `--model`           | `str` | 否  | `qwen2image_v1` | 通义千问文生图模型名称（示例：`qwen2image_v1`）   |
| `--size`            | `str` | 否  | `1024*1024`     | 生成图像分辨率，如 `512*512` 或 `1024*1024` |
| `--n`               | `int` | 否  | `1`             | 生成图像数量                            |
| `--seed`            | `int` | 否  | `None`          | 随机种子，用于结果复现                       |
| `--negative-prompt` | `str` | 否  | `None`          | 消极提示，用于排除不需要的视觉元素                 |
| `--prompt-extend`   | flag  | 否  | `False`         | 启用模型默认提示扩展                        |
| `--watermark`       | flag  | 否  | `False`         | 生成图像是否带水印                         |

---

### 四、使用示例

```bash
cd /root/UniMAG
source venv/bin/activate

# 设置通义千问 API Key
export QWEN_API_KEY="你的密钥"

# 运行 Graph2Image.py
python Graph2Image.py \
  --img-emb data/Grocery/image_features/node0_image.npy \
  --txt-emb data/Grocery/text_features/node0_text.npy \
  --top-k 12 \
  --model qwen2image_v1 \
  --size 1024*1024 \
  --n 1 \
  --seed 12345
```

**预期输出**：

```
已创建任务，task_id=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
[图1] URL: https://example.com/your_generated_image.png
```

---


