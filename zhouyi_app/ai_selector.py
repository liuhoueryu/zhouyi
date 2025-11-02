# -*- coding: utf-8 -*-
"""
AI分析器选择器
"""

import sys


def get_ai_analyzer():
    """获取可用的AI分析器"""

    # 首先尝试导入完整版AI分析器
    try:
        from .deep_learning import ai_analyzer
        print("使用完整版AI分析器")
        return ai_analyzer
    except Exception as e:
        print(f"完整版AI分析器不可用: {e}")

    # 如果完整版失败，创建基础版本
    print("使用基础功能AI分析器")
    return create_basic_analyzer()


def create_basic_analyzer():
    """创建基础功能AI分析器"""
    from .improved_sentiment import improved_sentiment_analyzer

    class BasicAIAnalyzer:
        def __init__(self):
            self.sentiment_analyzer = improved_sentiment_analyzer

        def analyze_sentiment(self, text):
            return self.sentiment_analyzer.analyze_sentiment(text)

        def predict_fortune_trend(self, hexagram_name, historical_data):
            return {"trend": "stable", "confidence": 0.5}

        def find_similar_cases(self, query, top_k=3):
            return []

        def generate_hexagram_interpretation(self, hexagram_name, user_thing, sentiment_analysis):
            sentiment = sentiment_analysis.get('sentiment', '中性')

            interpretations = {
                '积极': f"关于'{user_thing}'，{hexagram_name}卦象显示积极态势，宜把握时机。",
                '消极': f"面对'{user_thing}'的挑战，{hexagram_name}卦象建议谨慎行事。",
                '中性': f"关于'{user_thing}'，{hexagram_name}卦象显示需要更多观察。"
            }

            return interpretations.get(sentiment, interpretations['中性'])

    return BasicAIAnalyzer()


# 全局AI分析器实例
ai_analyzer = get_ai_analyzer()