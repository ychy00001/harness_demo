import os
import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_RETRY = int(os.getenv("MAX_RETRY", "3"))
MODEL_NAME = os.getenv("MODEL_NAME", "MiniMax-M2.7-highspeed")

OPENAI_BASE_URL="https://api.minimaxi.com/v1"
STATE_FILE = Path(__file__).parent / "task_state.json"

def create_client() -> OpenAI:
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        logger.error("未设置 MINIMAX_API_KEY 环境变量，请先设置后再运行")
        raise SystemExit(1)
    base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)

SYSTEM_PROMPT = """你是一个任务执行助手，具备以下能力：
1. 任务拆解：将复杂任务拆解为可执行的子任务
2. 命令执行：通过 execute_bash 工具执行 bash 命令
3. 状态更新：通过 update_task_status 工具更新任务状态

工作流程：
- 收到用户任务后，先拆解为子任务列表
- 逐个执行子任务，使用 execute_bash 执行需要的命令
- 每完成一个子任务，使用 update_task_status 更新状态
- 所有子任务完成后，报告完成
- 工作目录主要在当前目录下，子任务执行时需要在当前目录下执行

请始终用中文回复。"""

DECOMPOSE_PROMPT = """请将以下任务拆解为具体的子任务步骤，以 JSON 数组格式返回。
每个子任务包含 "id"（序号）、"description"（描述）、"status"（状态，初始为 "pending"）字段。
仅返回 JSON 数组，不要返回其他内容。

任务: {task}"""

VERIFY_PROMPT = """你是一个任务验证器。请根据以下信息判断任务是否正确完成。

原始任务: {task}

子任务执行记录:
{execution_log}

请判断任务是否正确完成，以 JSON 格式返回:
{{"completed": true/false, "reason": "判断理由", "remaining_tasks": ["未完成的任务描述1", "未完成的任务描述2"]}}

仅返回 JSON，不要返回其他内容。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "执行 bash 命令并返回输出结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 bash 命令",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task_status",
            "description": "更新子任务状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "子任务序号",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed"],
                        "description": "任务状态",
                    },
                    "result": {
                        "type": "string",
                        "description": "任务执行结果摘要",
                    },
                },
                "required": ["task_id", "status"],
            },
        },
    },
]

# 加载状态
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tasks": [], "execution_log": [], "created_at": None, "original_task": ""}

# 保存状态
def save_state(state: dict):
    state["updated_at"] = datetime.now().isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info("任务状态已保存到 %s", STATE_FILE)

# 执行 bash 命令
def execute_bash(command: str) -> str:
    logger.info("执行 bash 命令: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        if result.returncode != 0:
            logger.warning("命令返回非零退出码: %d", result.returncode)
        logger.info("命令输出: %s", output[:500])
        return output
    except subprocess.TimeoutExpired:
        logger.error("命令执行超时")
        return "错误: 命令执行超时(60s)"
    except Exception as e:
        logger.error("命令执行异常: %s", str(e))
        return f"错误: {str(e)}"

# 更新任务状态
def update_task_status(state: dict, task_id: int, status: str, result: str = "") -> str:
    for task in state["tasks"]:
        if task["id"] == task_id:
            task["status"] = status
            if result:
                task["result"] = result
            logger.info("子任务 #%d 状态更新: %s", task_id, status)
            save_state(state)
            return f"子任务 #{task_id} 状态已更新为 {status}"
    logger.warning("未找到子任务 #%d", task_id)
    return f"未找到子任务 #{task_id}"

# 任务拆解
def decompose_task(client: OpenAI, task: str) -> list[dict]:
    logger.info("开始拆解任务: %s", task)
    prompt = DECOMPOSE_PROMPT.format(task=task)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        extra_body={"reasoning_split": True},
    )
    content = response.choices[0].message.content.strip()
    logger.info("拆解原始响应: %s", content)

    json_str = content
    # 处理思考过程标签
    if "</think>" in content:
        json_str = content.split("</think>")[1].strip()
    # 处理代码块
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()

    subtasks = json.loads(json_str)
    logger.info("任务拆解完成，共 %d 个子任务", len(subtasks))
    for t in subtasks:
        logger.info("  - #%d: %s [%s]", t["id"], t["description"], t.get("status", "pending"))
    return subtasks

# 执行子任务
def execute_subtasks(client: OpenAI, state: dict) -> dict:
    logger.info("开始执行子任务...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请执行以下子任务:\n{json.dumps(state['tasks'], ensure_ascii=False, indent=2)}"},
    ]

    while True:
        pending_tasks = [t for t in state["tasks"] if t["status"] in ("pending", "in_progress")]
        if not pending_tasks:
            logger.info("所有子任务已执行完毕")
            break

        logger.info("剩余待执行子任务: %d", len(pending_tasks))
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
            # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
            extra_body={"reasoning_split": True},
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        if not assistant_msg.tool_calls:
            logger.info("模型未发起工具调用，执行结束")
            break

        messages.append(assistant_msg)

        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            logger.info("工具调用: %s(%s)", func_name, json.dumps(func_args, ensure_ascii=False))

            if func_name == "execute_bash":
                output = execute_bash(func_args["command"])
                state["execution_log"].append({
                    "type": "bash",
                    "command": func_args["command"],
                    "output": output,
                    "timestamp": datetime.now().isoformat(),
                })
                save_state(state)
            elif func_name == "update_task_status":
                output = update_task_status(
                    state,
                    func_args["task_id"],
                    func_args["status"],
                    func_args.get("result", ""),
                )
            else:
                output = f"未知工具: {func_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(output),
            })

    return state

# 验证任务完成情况
def verify_task(client: OpenAI, state: dict) -> dict:
    logger.info("开始验证任务完成情况...")
    execution_log_str = json.dumps(state["execution_log"], ensure_ascii=False, indent=2)
    tasks_str = json.dumps(state["tasks"], ensure_ascii=False, indent=2)
    full_log = f"子任务列表:\n{tasks_str}\n\n执行记录:\n{execution_log_str}"

    prompt = VERIFY_PROMPT.format(task=state["original_task"], execution_log=full_log)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        extra_body={"reasoning_split": True},
    )
    content = response.choices[0].message.content.strip()
    # logger.info("验证原始响应: %s", content)

    json_str = content
    # 处理思考过程标签
    if "</think>" in content:
        json_str = content.split("</think>")[1].strip()
    # 处理代码块
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()

    result = json.loads(json_str)
    logger.info("验证结果: completed=%s, reason=%s", result.get("completed"), result.get("reason"))
    return result


def main():
    logger.info("=" * 60)
    logger.info("智能体2 启动")
    logger.info("=" * 60)

    client = create_client()

    user_input = input("请输入你的任务: ").strip()
    if not user_input:
        logger.warning("输入为空，退出程序")
        return

    state = load_state()
    state["original_task"] = user_input
    state["created_at"] = datetime.now().isoformat()
    state["execution_log"] = []
    save_state(state)

    retry_count = 0
    while retry_count < MAX_RETRY:
        retry_count += 1
        logger.info("=" * 60)
        logger.info("第 %d 轮执行 (最多 %d 轮)", retry_count, MAX_RETRY)
        logger.info("=" * 60)

        pending_or_failed = [
            t for t in state["tasks"]
            if t.get("status") in ("pending", "in_progress", "failed")
        ]

        if not pending_or_failed or retry_count == 1:
            logger.info("拆解任务...")
            subtasks = decompose_task(client, user_input)
            state["tasks"] = subtasks
            save_state(state)
        else:
            logger.info("存在未完成的子任务，继续执行...")
            for t in state["tasks"]:
                if t["status"] == "failed":
                    t["status"] = "pending"
            save_state(state)

        state = execute_subtasks(client, state)

        verify_result = verify_task(client, state)
        if verify_result.get("completed", False):
            logger.info("=" * 60)
            logger.info("任务验证通过！任务已完成！")
            logger.info("验证理由: %s", verify_result.get("reason", ""))
            logger.info("=" * 60)
            break
        else:
            logger.warning("任务验证未通过: %s", verify_result.get("reason", ""))
            remaining = verify_result.get("remaining_tasks", [])
            if remaining:
                logger.info("未完成的任务: %s", json.dumps(remaining, ensure_ascii=False))
                max_id = max(t["id"] for t in state["tasks"]) if state["tasks"] else 0
                for i, desc in enumerate(remaining, 1):
                    state["tasks"].append({
                        "id": max_id + i,
                        "description": desc,
                        "status": "pending",
                    })
                save_state(state)

            if retry_count >= MAX_RETRY:
                logger.error("已达最大重试次数 (%d)，任务未完成", MAX_RETRY)
                break
            logger.info("将重新执行未完成的子任务...")

    logger.info("=" * 60)
    logger.info("最终任务状态:")
    for t in state["tasks"]:
        logger.info("  - #%d: %s [%s]", t["id"], t["description"], t["status"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
