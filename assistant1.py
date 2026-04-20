import os
import logging
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "MiniMax-M2.5")

OPENAI_BASE_URL="https://api.minimaxi.com/v1"

def create_client() -> OpenAI:
    api_key = os.getenv("OPENAI_AMINIMAX_API_KEYPI_KEY")
    if not api_key:
        logger.error("未设置 MINIMAX_API_KEY 环境变量，请先设置后再运行")
        raise SystemExit(1)
    base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def chat(client: OpenAI, user_input: str) -> str:
    logger.info("发送请求到 LLM，模型: %s", MODEL_NAME)
    logger.info("用户输入: %s", user_input)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手，请用中文回答问题。"},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
    )

    result = response.choices[0].message.content
    logger.info("LLM 响应: %s", result)
    return result


def main():
    logger.info("=" * 50)
    logger.info("AI 助手1 启动")
    logger.info("=" * 50)

    client = create_client()

    user_input = input("请输入你的问题: ").strip()
    if not user_input:
        logger.warning("输入为空，退出程序")
        return

    logger.info("开始处理用户请求...")
    result = chat(client, user_input)
    logger.info("=" * 50)
    logger.info("最终结果: %s", result)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
