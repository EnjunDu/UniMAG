import os
import time
import requests

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
    return data["output"]["task_id"]  # PENDING 状态 :contentReference[oaicite:2]{index=2}

# ======================
# 步骤2：轮询查询结果
# ======================
def fetch_results(task_id: str, interval: int = 5, timeout: int = 300):
    """
    根据 task_id 反复查询，直到 SUCCEEDED 或 FAILED
    文档示例：GET /api/v1/tasks/{task_id} :contentReference[oaicite:3]{index=3}
    """
    url = f"{BASE_URL}/tasks/{task_id}"
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(url, headers={"Authorization": f"Bearer {API_KEY}"})
        resp.raise_for_status()
        result = resp.json()["output"]
        status = result["task_status"]
        if status == "SUCCEEDED":
            return result["results"]  # 包含 URL 列表
        elif status in ("FAILED", "CANCELED"):
            raise RuntimeError(f"任务 {task_id} 失败，原因：{result.get('code')} / {result.get('message')}")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"等待任务 {task_id} 超时，已用 {timeout} 秒")

# ======================
# 主流程示例
# ======================
if __name__ == "__main__":
    # 1. 发起任务
    task_id = create_task(
        prompt="根据以下多模态embeding生图：",
        negative_prompt="无",
        size="1024*1024",
        n=1,
        seed=12345,
        prompt_extend=True,
        watermark=False,
    )
    print(f"已创建任务，task_id={task_id}")

    # 2. 轮询并获取结果
    try:
        results = fetch_results(task_id)
        for idx, item in enumerate(results, 1):
            print(f"[图{idx}] URL: {item['url']}")
    except Exception as e:
        print("查询失败：", e)
