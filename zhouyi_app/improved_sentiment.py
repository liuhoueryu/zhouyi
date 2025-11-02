# -*- coding: utf-8 -*-
"""
改进的情感分析模块
"""

import jieba
import re
from collections import Counter


class ImprovedSentimentAnalyzer:
    def __init__(self):
        # 加载情感词典
        self.positive_dict = self.load_sentiment_dict('positive')
        self.negative_dict = self.load_sentiment_dict('negative')
        self.degree_dict = self.load_degree_dict()
        self.negation_dict = self.load_negation_dict()

        # 初始化jieba，添加情感词汇
        for word in self.positive_dict:
            jieba.add_word(word)
        for word in self.negative_dict:
            jieba.add_word(word)

    def load_sentiment_dict(self, sentiment_type):
        """加载情感词典"""
        if sentiment_type == 'positive':
            return {
                '开心': 2, '高兴': 2, '快乐': 2, '喜悦': 2, '幸福': 3,
                '顺利': 1, '成功': 2, '好运': 2, '满意': 1, '喜欢': 1,
                '爱': 3, '希望': 1, '期待': 1, '兴奋': 2, '激动': 1,
                '美好': 1, '完美': 2, '优秀': 1, '精彩': 1, '厉害': 1,
                '感谢': 1, '感动': 2, '温暖': 1, '安心': 1, '放松': 1,
                '充满信心': 2, '乐观': 2, '积极': 2, '向上': 1, '庆祝': 1,
                '胜利': 2, '成就': 1, '进步': 1, '发展': 1, '提升': 1,
                '健康': 1, '平安': 1, '吉祥': 1, '如意': 1, '心想事成': 2
            }
        else:  # negative
            return {
                '伤心': 2, '难过': 2, '痛苦': 3, '悲伤': 2, '绝望': 3,
                '困难': 1, '失败': 2, '问题': 1, '担心': 1, '焦虑': 2,
                '害怕': 2, '恐惧': 3, '紧张': 1, '压力': 1, '烦恼': 1,
                '生气': 2, '愤怒': 3, '失望': 2, '沮丧': 2, '郁闷': 1,
                '讨厌': 1, '恨': 3, '后悔': 1, '愧疚': 1, '自责': 1,
                '无助': 2, '孤独': 2, '寂寞': 1, '疲惫': 1, '累': 1,
                '痛苦': 2, '折磨': 2, '煎熬': 2, '困境': 1, '危机': 2,
                '损失': 1, '伤害': 2, '痛苦': 2, '病痛': 2, '死亡': 3
            }

    def load_degree_dict(self):
        """加载程度副词词典"""
        return {
            '非常': 1.5, '很': 1.3, '特别': 1.4, '极其': 1.6, '极度': 1.6,
            '相当': 1.2, '比较': 1.1, '稍微': 0.8, '有点': 0.8, '略微': 0.8,
            '十分': 1.4, '格外': 1.3, '异常': 1.5, '超级': 1.4, '巨': 1.4
        }

    def load_negation_dict(self):
        """加载否定词词典"""
        return {
            '不', '没', '没有', '无', '未', '别', '莫', '勿', '非', '否'
        }

    def analyze_sentiment(self, text):
        """改进的情感分析"""
        if not text or not str(text).strip():
            return {"sentiment": "中性", "confidence": 0.5}

        # 分词
        words = list(jieba.cut(str(text)))

        positive_score = 0
        negative_score = 0
        current_degree = 1.0
        negation_count = 0

        for i, word in enumerate(words):
            # 检查程度副词
            if word in self.degree_dict:
                current_degree = self.degree_dict[word]
                continue

            # 检查否定词
            if word in self.negation_dict:
                negation_count += 1
                continue

            # 计算情感分数
            if word in self.positive_dict:
                score = self.positive_dict[word] * current_degree
                if negation_count % 2 == 1:  # 奇数个否定词，情感反转
                    negative_score += score
                else:
                    positive_score += score
                current_degree = 1.0
                negation_count = 0

            elif word in self.negative_dict:
                score = self.negative_dict[word] * current_degree
                if negation_count % 2 == 1:  # 奇数个否定词，情感反转
                    positive_score += score
                else:
                    negative_score += score
                current_degree = 1.0
                negation_count = 0

        # 计算总得分和置信度
        total_score = positive_score + negative_score

        if total_score == 0:
            # 没有情感词，检查情感倾向词
            return self.fallback_analysis(text)

        # 归一化得分
        max_score = max(self.positive_dict.values()) * 3  # 假设最大可能得分

        if positive_score > negative_score:
            sentiment = "积极"
            confidence = min(positive_score / max_score, 0.95)
        elif negative_score > positive_score:
            sentiment = "消极"
            confidence = min(negative_score / max_score, 0.95)
        else:
            sentiment = "中性"
            confidence = 0.5

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "model_used": "improved_keyword"
        }

    def fallback_analysis(self, text):
        """备用分析方法"""
        text_lower = str(text).lower()

        # 检查疑问词
        question_words = ['吗', '么', '如何', '怎样', '为什么', '为何', '能不能']
        if any(word in text_lower for word in question_words):
            return {"sentiment": "中性", "confidence": 0.6, "reason": "疑问句"}

        # 检查祈使句
        imperative_indicators = ['请', '希望', '建议', '应该', '要']
        if any(word in text_lower for word in imperative_indicators):
            return {"sentiment": "中性", "confidence": 0.6, "reason": "祈使句"}

        # 默认中性
        return {"sentiment": "中性", "confidence": 0.5, "reason": "无明确情感词"}


# 全局改进的情感分析器实例
improved_sentiment_analyzer = ImprovedSentimentAnalyzer()