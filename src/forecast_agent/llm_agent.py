from typing import List, Dict, Optional
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .config import Settings


def _normalize_base_url(url: str | None) -> str | None:
    """
    DeepSeek/OpenAI 兼容接口要求 base_url 为根域名，如 https://api.deepseek.com
    若用户传入 /v1/chat/completions 等完整路径，需截断到根，避免 404。
    """
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    # 仅保留 scheme+netloc
    return f"{parsed.scheme}://{parsed.netloc}"


def analyze_forecast_with_llm(
    metrics: List[Dict],
    context: str,
    settings: Settings,
    max_tokens: int = 600,
) -> Optional[str]:
    """
    使用 LangChain + DeepSeek (OpenAI 兼容接口) 生成结果分析。
    """
    if not settings.deepseek_api_key or not settings.deepseek_api_url:
        return None

    try:
        base_url = _normalize_base_url(settings.deepseek_api_url)
        llm = ChatOpenAI(
            model=settings.model_name,
            base_url=base_url,
            api_key=settings.deepseek_api_key,
            temperature=0.2,
            max_tokens=max_tokens,
        )

        prompt = (
            "你是一名财务数据分析专家，擅长结合财务损益数据,利用算法模型对未来的损益进行预测,请基于以下时间序列预测结果进行分析，"
            "给出模型优缺点、潜在改进方向，并用中文总结：\n"
            f"评估指标(JSON): {metrics}\n"
            f"补充上下文: {context}\n"
        )

        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content if response else None
    except Exception as exc:  # noqa: BLE001
        # 避免 LLM 请求失败导致整条链路报错
        print(f"LLM 调用失败: {exc}")
        return None

