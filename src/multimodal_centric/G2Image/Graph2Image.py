#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import requests
import argparse
import numpy as np

# ======================
# 配置部分
# ======================
BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 请先导出：export DASHSCOPE_API_KEY=你的密钥
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "X-DashScope-Async": "enable",      # 必选：启用异步处理
}

# ======================
# 步骤1：创建任务
# ======================
def create_task(
    prompt: str,
    model: str = "wanx2.1-t2i-turbo",
    size: str = "1024*1024",
    n: int = 1,
    negative_prompt: str = None,
    seed: int = None,
    prompt_extend: bool = True,
    watermark: bool = False,
) -> str:
    url = f"{BASE_URL}/services/aigc/text2image/image-synthesis"
    body = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {"size": size, "n": n, "prompt_extend": prompt_extend, "watermark": watermark},
    }
    if negative_prompt:
        body["input"]["negative_prompt"] = negative_prompt
    if seed is not None:
        body["parameters"]["seed"] = seed

    resp = requests.post(url, headers=HEADERS, json=body)
    resp.raise_for_status()
    data = resp.json()
    return data["output"]["task_id"]

# ======================
# 步骤2：轮询查询结果
# ======================
def fetch_results(task_id: str, interval: int = 5, timeout: int = 300):
    url = f"{BASE_URL}/tasks/{task_id}"
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(url, headers={"Authorization": f"Bearer {API_KEY}"})
        resp.raise_for_status()
        result = resp.json()["output"]
        status = result["task_status"]
        if status == "SUCCEEDED":
            return result["results"]
        elif status in ("FAILED", "CANCELED"):
            raise RuntimeError(f"任务 {task_id} 失败，原因：{result.get('code')} / {result.get('message')}")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"等待任务 {task_id} 超时，已用 {timeout} 秒")

# ======================
# 主流程：接收已处理的嵌入并生成图像
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="基于已处理的图像和文本模态嵌入直接生成图像描述"
    )
    parser.add_argument("--img-emb",     type=str, required=True, help="图像嵌入文件路径 (.npy)")
    parser.add_argument("--txt-emb",     type=str, required=True, help="文本嵌入文件路径 (.npy)")
    parser.add_argument("--top-k",       type=int, default=10, help="提示中展示前 K 维特征")
    parser.add_argument("--model",       type=str, default="wanx2.1-t2i-turbo")
    parser.add_argument("--size",        type=str, default="1024*1024")
    parser.add_argument("--n",           type=int, default=1)
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--prompt-extend",   action="store_true", help="启用模型默认提示扩展")
    parser.add_argument("--watermark",       action="store_true", help="输出图像是否带水印")
    args = parser.parse_args()

    # 加载嵌入向量
    img_vec = np.load(args.img_emb)
    txt_vec = np.load(args.txt_emb)
    top_k  = args.top_k
    img_list = img_vec[:top_k].tolist()
    txt_list = txt_vec[:top_k].tolist()

    # 构建 prompt
    prompt = f"""
你正在观察一个多模态属性图中的节点。以下是该节点的前{top_k}维模态特征：

图像模态特征: {img_list}
文本模态特征: {txt_list}

请结合这些特征，推测该节点可能代表的内容，然后生成一张图片描述目标节点。无需给出推理过程。
"""

    # 发起任务
    task_id = create_task(
        prompt=prompt,
        model=args.model,
        size=args.size,
        n=args.n,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        prompt_extend=args.prompt_extend,
        watermark=args.watermark,
    )
    print(f"已创建任务，task_id={task_id}")

    # 查询并打印结果
    try:
        results = fetch_results(task_id)
        for idx, item in enumerate(results, 1):
            print(f"[图{idx}] URL: {item['url']}")
    except Exception as e:
        print("查询失败：", e)
