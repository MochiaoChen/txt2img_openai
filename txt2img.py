"""
text2img_openai.py
------------------
A tiny helper for generating images from text via the OpenAI API.
Compatible with GPT-4o-image 及 DALL·E-3 / DALL·E-2.

准备：
1. pip install --upgrade openai pillow matplotlib
2. 在系统环境中设置 OPENAI_API_KEY
   （如在受限地区，可额外指定 base_url 参数走代理）
"""

import os
import time
import base64
import io
from pathlib import Path
from typing import Literal, Optional

import openai
from PIL import Image
import matplotlib.pyplot as plt

# ---------- 基础配置 ----------
# 自动读取环境变量中的 OPENAI_API_KEY
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # 若位于受限网络，可将下一行取消注释并替换为代理地址
    # base_url="https://api.your-proxy-domain.com/v1"
)

# ---------- 核心函数 ----------
def generate_image(
    prompt: str,
    model: Literal[
        "gpt-4o-image-alpha",  # GPT-4o（2025）图像模型名称可能随版本略有变化
        "dall-e-3",
        "dall-e-2",
    ] = "dall-e-3",
    size: Literal["256x256", "512x512", "1024x1024"] = "1024x1024",
    fmt: Literal["url", "b64_json"] = "url",
    retries: int = 3,
):
    """
    文生图：返回图片 URL 或 Base64 数据。

    参数
    ----
    prompt  : 描述图片内容的中文/英文提示词
    model   : 使用的模型名称
    size    : 输出分辨率
    fmt     : 'url' → 返回公网 URL
              'b64_json' → 返回 base64 字符串，适合离线保存
    retries : 自动重试次数(RateLimit 或网络错误）
    """
    for attempt in range(retries):
        try:
            # OpenAI SDK ≥1.12 推荐的新写法，thread-safe
            # GPT-4o 使用 chat.completions + modalities；DALL·E 用 images.generate
            if model.startswith("gpt-4o"):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    modalities=["text", "image"],
                    max_tokens=0,              # 生成纯图片时可设为 0
                    response_format={"image": fmt},
                    # top_p/temperature 在纯文生图时通常不用设置
                )
                img_data = response.choices[0].message.image
            else:
                response = client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    size=size,
                    response_format=fmt,
                )
                img_data = response.data[0]

            # 返回 URL 或 Base64
            return img_data.url if fmt == "url" else img_data.b64_json

        except openai.RateLimitError as e:
            wait = 2 ** attempt + 0.5
            print(f"[RateLimit] 第 {attempt+1}/{retries} 次重试，等待 {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}") from e

    raise TimeoutError("已达到最大重试次数")


# ---------- 示例用法 ----------
if __name__ == "__main__":
    demo_prompt = (
        "A photorealistic image of a futuristic city at golden hour, flying cars, "
        "ultra-detailed skyscrapers, cinematic lighting"
    )

    # 1) 获取公网 URL
    url = generate_image(demo_prompt, model="dall-e-3", fmt="url")
    print("Image URL →", url)

    # 2) 获取 Base64 并保存到本地
    b64 = generate_image(demo_prompt, model="dall-e-3", fmt="b64_json")
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    outfile = Path("demo.png")
    img.save(outfile)
    print("Saved to", outfile.resolve())

    # 3) Inline 显示
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
