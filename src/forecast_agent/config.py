import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    全局配置，使用环境变量驱动。
    """

    data_path: Path = Field(
        default=Path("data/sample_data.xlsx"),
        description="Excel 数据文件路径，要求包含日期列与数值列。",
    )
    date_column: str = Field(default="date", description="日期列名，需可解析为 datetime。")
    target_column: str = Field(default="value", description="目标数值列名。")
    output_dir: Path = Field(default=Path("outputs"), description="结果输出目录。")

    # LLM / DeepSeek
    deepseek_api_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_URL"),
        description="DeepSeek 兼容的 Chat Completions API URL。",
    )
    deepseek_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"),
        description="DeepSeek API Key。",
    )
    model_name: str = Field(default="deepseek-chat", description="LLM 模型名称。")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_settings(env_file: str | None = ".env") -> Settings:
    """
    先加载 .env，再返回 Settings。
    """
    if env_file:
        load_dotenv(env_file)
    return Settings()

