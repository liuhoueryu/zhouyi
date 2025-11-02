# -*- coding: utf-8 -*-
"""
情感分析诊断脚本
"""

import sys
import os

# 添加当前路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fortune_teller.settings')

import django

django.setup()

from zhouyi_app.deep_learning import ai_analyzer


def test_sentiment_analysis():
    """测试情感分析功能"""
    print("=== 情感分析诊断 ===")

    # 测试用例
    test_cases = [
        "我今天非常开心，一切顺利",
        "我很担心明天的考试，感觉很焦虑",
        "这件事让我很生气",
        "我对未来充满希望",
        "我感到很失望",
        "这个结果让我很满意",
        "我害怕面对这个挑战",
        "我很兴奋能够参与这个项目"
    ]

    print(f"情感分析器状态: {ai_analyzer.sentiment_analyzer is not None}")

    if ai_analyzer.sentiment_analyzer:
        print("情感分析模型已加载")
        # 检查模型信息
        try:
            model_name = ai_analyzer.sentiment_analyzer.model.name_or_path
            print(f"模型名称: {model_name}")
        except:
            print("无法获取模型信息")
    else:
        print("情感分析模型未加载，使用备用方案")

    print("\n=== 测试结果 ===")

    for i, text in enumerate(test_cases, 1):
        result = ai_analyzer.analyze_sentiment(text)
        print(f"{i}. '{text}'")
        print(f"   情感: {result['sentiment']}, 置信度: {result['confidence']:.3f}")
        print(f"   使用模型: {result.get('model_used', 'unknown')}")
        print()


def check_models():
    """检查模型状态"""
    print("=== 模型状态检查 ===")
    print(f"Transformers可用: {ai_analyzer.versions.get('transformers_available', False)}")
    print(f"情感分析器: {ai_analyzer.sentiment_analyzer is not None}")
    print(f"文本生成器: {ai_analyzer.text_generator is not None}")


if __name__ == "__main__":
    test_sentiment_analysis()
    check_models()