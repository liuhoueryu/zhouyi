# -*- coding: utf-8 -*-
"""
Ollama大模型集成模块
使用qwen3:latest模型生成个性化的卦象解释
"""

import requests
import json
import time
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url="http://192.168.124.3:11434", model="qwen3:latest", timeout=30):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.available = self.check_availability()

    def check_availability(self):
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama服务可用")
                # 检查模型是否存在
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                if any(self.model in name for name in model_names):
                    logger.info(f"模型 {self.model} 可用")
                    return True
                else:
                    logger.warning(f"模型 {self.model} 未找到，可用的模型: {model_names}")
            return False
        except Exception as e:
            logger.warning(f"Ollama服务不可用: {e}")
            return False

    def generate_interpretation(self, hexagram_name, user_thing, sentiment_analysis, changing_lines=None):
        """使用Ollama生成个性化的卦象解释"""
        if not self.available:
            raise Exception("Ollama服务不可用")

        # 构建提示词
        prompt = self.build_prompt(hexagram_name, user_thing, sentiment_analysis, changing_lines)

        try:
            # 调用Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                interpretation = result.get('response', '').strip()

                # 清理响应内容
                interpretation = self.clean_response(interpretation)
                return interpretation
            else:
                logger.error(f"Ollama API错误: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API返回错误: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.error("Ollama请求超时")
            raise Exception("生成解释超时，请稍后重试")
        except Exception as e:
            logger.error(f"Ollama调用失败: {e}")
            raise Exception("生成解释失败")

    def build_prompt(self, hexagram_name, user_thing, sentiment_analysis, changing_lines=None):
        """构建提示词"""
        sentiment = sentiment_analysis.get('sentiment', '中性')
        confidence = sentiment_analysis.get('confidence', 0.5)

        # 变爻信息
        changing_info = ""
        if changing_lines and len(changing_lines) > 0:
            changing_lines_str = "、".join([f"第{line + 1}爻" for line in changing_lines])
            changing_info = f"本卦有变爻出现在{changing_lines_str}，需要结合变卦来综合解读。"

        prompt = f"""你是一位资深的周易大师，请根据以下信息为用户生成个性化的卦象解释：

卦象名称：{hexagram_name}
用户询问事项：{user_thing}
用户情感倾向：{sentiment}（置信度：{confidence:.2f}）
{changing_info}

请根据周易{hexagram_name}卦的卦辞、爻辞和传统解释，结合用户的具体事项和情感状态，生成一段富有智慧和启发性的解读。

要求：
1. 解释要贴近用户的具体情况"{user_thing}"
2. 考虑用户的情感状态"{sentiment}"
3. 提供实用的建议和启示
4. 语言要优美、富有哲理，但避免过于晦涩
5. 长度在200-300字左右
6. 以第二人称"您"来称呼用户

请开始你的解读："""

        return prompt

    def clean_response(self, text):
        """清理模型响应"""
        # 移除可能的提示词残留
        clean_text = text
        if "请开始你的解读：" in clean_text:
            clean_text = clean_text.split("请开始你的解读：")[-1]

        # 移除开头的空行和空格
        clean_text = clean_text.strip()

        return clean_text


# 全局Ollama客户端实例
try:
    ollama_client = OllamaClient(model="qwen3:latest")  # 根据你的实际模型名称调整
except Exception as e:
    logger.error(f"初始化Ollama客户端失败: {e}")
    ollama_client = None