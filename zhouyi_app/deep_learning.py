# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys
import os

# 设置标准输出编码为UTF-8，解决Windows编码问题
try:
    if sys.platform == "win32":
        # Windows系统下的编码修复
        sys.stdout.reconfigure(encoding='utf-8')
except:
    pass


def safe_print(message):
    """安全的打印函数，避免编码错误"""
    try:
        # 替换Unicode表情为文本描述
        replacements = {
            '🤖': '[AI]',
            '📊': '[DATA]',
            '🔥': '[TORCH]',
            '🔧': '[TOOL]',
            '🎯': '[TARGET]',
            '🔄': '[PROCESS]',
            '❌': '[ERROR]',
            '✅': '[OK]',
            '⚠️': '[WARN]',
            '🎉': '[SUCCESS]',
            '🤔': '[THINK]',
            '💭': '[IDEA]',
            '🌱': '[GROW]',
            '🛡️': '[PROTECT]',
            '🔍': '[SEARCH]',
            '🚀': '[LAUNCH]',
            '📈': '[TREND_UP]',
            '⚖️': '[BALANCE]'
        }

        clean_message = message
        for emoji, text in replacements.items():
            clean_message = clean_message.replace(emoji, text)

        print(clean_message)
    except UnicodeEncodeError:
        # 如果还有编码错误，使用ASCII安全的输出
        ascii_message = message.encode('utf-8', 'ignore').decode('utf-8')
        print(ascii_message)


# 安全导入TensorFlow
try:
    import tensorflow as tf

    TF_AVAILABLE = True
    TF_VERSION = tf.__version__
    safe_print("[OK] TensorFlow " + TF_VERSION + " Loaded successfully")
except ImportError as e:
    safe_print("[ERROR] TensorFlow is Unuseful: " + str(e))
    TF_AVAILABLE = False
    TF_VERSION = "未安装"

# 安全导入PyTorch和Transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch

    TRANSFORMERS_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    safe_print("[OK] PyTorch " + TORCH_VERSION + " Loaded successfully")
except ImportError as e:
    safe_print("[ERROR] PyTorch/Transformers Unuseful: " + str(e))
    TRANSFORMERS_AVAILABLE = False
    TORCH_VERSION = "未安装"
try:
    from .ollama_integration import ollama_client

    OLLAMA_AVAILABLE = ollama_client is not None and ollama_client.available
    if OLLAMA_AVAILABLE:
        safe_print("Ollama successful")
    else:
        safe_print("Ollama unuseful")
except ImportError as e:
    safe_print(f"Ollama failure: {e}")
    OLLAMA_AVAILABLE = False


class FortuneAIAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        self.text_generator = None
        self.similarity_model = None
        self.vectorizer = TfidfVectorizer()
        self.similarity_matrix = None
        self.record_texts = []
        self.records = []

        self.versions = {
            'torch': TORCH_VERSION,
            'transformers_available': TRANSFORMERS_AVAILABLE
        }

        self.initialize_models()

    def initialize_models(self):
        """初始化深度学习模型"""
        safe_print("初始化AI模型...")

        # 情感分析模型 - 使用更好的中文模型
        if TRANSFORMERS_AVAILABLE:
            try:
                # 尝试加载专门的中文情感分析模型
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    framework="pt"
                )
                safe_print("中文情感分析模型加载成功 (seamew/roberta-wwm-chinese-text-classification)")
            except Exception as e:
                safe_print(f"中文情感分析模型加载失败: {e}")
                try:
                    # 备用模型1
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                        framework="pt"
                    )
                    safe_print("多语言情感分析模型加载成功 (twitter-xlm-roberta-base-sentiment)")
                except Exception as e2:
                    safe_print(f"多语言情感分析模型加载失败: {e2}")
                    try:
                        # 备用模型2 - 使用更小的模型
                        self.sentiment_analyzer = pipeline(
                            "sentiment-analysis",
                            framework="pt"
                        )
                        safe_print("默认情感分析模型加载成功")
                    except Exception as e3:
                        safe_print(f"所有情感分析模型加载失败: {e3}")
                        self.sentiment_analyzer = None
        else:
            safe_print("Transformers不可用，跳过情感分析模型")

        safe_print("AI模型初始化完成")

    def analyze_sentiment(self, text):
        """分析用户输入的情感倾向 - 改进版本"""
        if not text or not str(text).strip():
            return {
                "sentiment": "中性",
                "confidence": 0.5,
                "model_used": "fallback_empty_text"
            }

        # 如果模型不可用，使用基于关键词的分析
        if not self.sentiment_analyzer:
            return self.keyword_based_sentiment_analysis(text)

        try:
            # 清理文本
            clean_text = str(text).strip()[:500]

            # 执行情感分析
            result = self.sentiment_analyzer(clean_text)

            sentiment = result[0]['label']
            confidence = result[0]['score']

            safe_print(f"情感分析原始结果: {sentiment}, 置信度: {confidence}")

            # 改进的标签映射
            sentiment_map = {
                # 二分类模型
                'LABEL_0': '消极',
                'LABEL_1': '积极',
                'negative': '消极',
                'positive': '积极',
                # 三分类模型
                'LABEL_2': '中性',  # 有些模型的中性标签
                'neutral': '中性',
                # 多语言模型
                'Negative': '消极',
                'Positive': '积极',
                'Neutral': '中性',
                # 星级评分模型
                '1 star': '消极',
                '2 stars': '消极',
                '3 stars': '中性',
                '4 stars': '积极',
                '5 stars': '积极'
            }

            detected_sentiment = sentiment_map.get(sentiment.lower(), '中性')

            # 根据置信度调整结果
            if confidence < 0.4:
                # 置信度太低，使用关键词分析
                keyword_result = self.keyword_based_sentiment_analysis(text)
                if keyword_result['confidence'] > 0.6:
                    return keyword_result

            # 特殊处理：如果置信度中等但结果与关键词分析不一致，使用关键词分析
            if 0.4 <= confidence < 0.7:
                keyword_result = self.keyword_based_sentiment_analysis(text)
                if (keyword_result['sentiment'] != detected_sentiment and
                        keyword_result['confidence'] > 0.6):
                    safe_print(f"模型与关键词分析不一致，使用关键词结果")
                    return keyword_result

            return {
                "sentiment": detected_sentiment,
                "confidence": float(confidence),
                "raw_sentiment": sentiment,
                "model_used": "transformer_model"
            }

        except Exception as e:
            safe_print(f"情感分析失败: {e}")
            # 回退到关键词分析
            return self.keyword_based_sentiment_analysis(text)

    def keyword_based_sentiment_analysis(self, text):
        """基于关键词的情感分析（备用方案）"""
        if not text:
            return {"sentiment": "中性", "confidence": 0.5, "model_used": "keyword_fallback"}

        text_lower = str(text).lower()

        # 扩展的情感关键词库
        positive_words = {
            '开心': 2, '高兴': 2, '快乐': 2, '喜悦': 2, '幸福': 2,
            '顺利': 1, '成功': 1, '好运': 1, '满意': 1, '喜欢': 1,
            '爱': 2, '希望': 1, '期待': 1, '兴奋': 2, '激动': 1,
            '美好': 1, '完美': 1, '优秀': 1, '精彩': 1, '厉害': 1,
            '感谢': 1, '感动': 1, '温暖': 1, '安心': 1, '放松': 1,
            '充满信心': 2, '乐观': 2, '积极': 2, '向上': 1
        }

        negative_words = {
            '伤心': 2, '难过': 2, '痛苦': 2, '悲伤': 2, '绝望': 3,
            '困难': 1, '失败': 1, '问题': 1, '担心': 1, '焦虑': 2,
            '害怕': 2, '恐惧': 2, '紧张': 1, '压力': 1, '烦恼': 1,
            '生气': 2, '愤怒': 2, '失望': 2, '沮丧': 2, '郁闷': 1,
            '讨厌': 1, '恨': 2, '后悔': 1, '愧疚': 1, '自责': 1,
            '无助': 2, '孤独': 2, '寂寞': 1, '疲惫': 1, '累': 1
        }

        # 计算情感分数
        positive_score = 0
        negative_score = 0

        for word, weight in positive_words.items():
            if word in text_lower:
                positive_score += weight

        for word, weight in negative_words.items():
            if word in text_lower:
                negative_score += weight

        # 决定情感倾向
        total_score = positive_score + negative_score

        if total_score == 0:
            return {"sentiment": "中性", "confidence": 0.5, "model_used": "keyword_no_match"}

        # 计算置信度
        max_possible_score = max(
            sum(positive_words.values()),
            sum(negative_words.values())
        ) / 10  # 归一化

        confidence = min(total_score / max_possible_score, 0.9)

        if positive_score > negative_score:
            sentiment = "积极"
            final_confidence = max(0.6, confidence)
        elif negative_score > positive_score:
            sentiment = "消极"
            final_confidence = max(0.6, confidence)
        else:
            sentiment = "中性"
            final_confidence = 0.5

        safe_print(f"关键词分析: 积极{positive_score}, 消极{negative_score}, 情感{sentiment}")

        return {
            "sentiment": sentiment,
            "confidence": final_confidence,
            "model_used": "keyword_based"
        }

    # 其他方法保持不变...
    def predict_fortune_trend(self, hexagram_name, historical_data):
        """预测卦象趋势（基于历史数据）"""
        if not historical_data:
            return {"trend": "stable", "confidence": 0.5}

        try:
            # 简单的趋势分析
            hexagram_counts = {}
            for record in historical_data:
                name = record.hexagram_name
                hexagram_counts[name] = hexagram_counts.get(name, 0) + 1

            total = len(historical_data)
            current_hexagram_count = hexagram_counts.get(hexagram_name, 0)
            probability = current_hexagram_count / total if total > 0 else 0

            # 基于概率判断趋势
            if probability > 0.3:
                trend = "rising"
                confidence = min(probability * 2, 0.9)
            elif probability > 0.1:
                trend = "stable"
                confidence = 0.6
            else:
                trend = "emerging"
                confidence = 0.7

            return {
                "trend": trend,
                "confidence": float(confidence),
                "probability": float(probability),
                "total_cases": total
            }
        except Exception as e:
            safe_print(f"趋势预测失败: {e}")
            return {"trend": "stable", "confidence": 0.5}

    def safe_array_check(self, array, threshold=0.1):
        """安全地检查数组条件，避免布尔值歧义"""
        if array is None or len(array) == 0:
            return False

        # 如果是标量，直接比较
        if np.isscalar(array):
            return array > threshold

        # 如果是数组，使用any()或all()
        try:
            # 对于相似度数组，我们关心是否有任何值超过阈值
            return (array > threshold).any()
        except ValueError as e:
            safe_print("[ERROR] 数组检查错误: " + str(e))
            return False

    def train_similarity_model(self, records):
        """训练相似度匹配模型（使用安全的数组操作）"""
        if not records:
            return

        try:
            # 准备训练数据
            texts = []
            self.records = list(records)[:100]  # 限制数量避免内存问题

            for record in self.records:
                text = f"{record.thing} {record.hexagram_name}"
                texts.append(self.preprocess_text(text))

            # 训练TF-IDF模型
            if texts:
                self.similarity_matrix = self.vectorizer.fit_transform(texts)
                self.record_texts = texts
                safe_print("[OK] similarity model Training successful，共 " + str(len(texts)) + " 条记录")
        except Exception as e:
            safe_print("[ERROR] similarity model Training failure: " + str(e))

    # def find_similar_cases(self, query, top_k=5):
    #     """查找相似的历史案例（修复数组布尔判断）"""
    #     if (self.similarity_matrix is None or
    #             not hasattr(self, 'records') or
    #             not self.records):
    #         return []
    #
    #     try:
    #         # 预处理查询文本
    #         processed_query = self.preprocess_text(query)
    #         query_vector = self.vectorizer.transform([processed_query])
    #
    #         # 计算相似度 - 使用安全的数组操作
    #         similarity_scores = cosine_similarity(query_vector, self.similarity_matrix)
    #
    #         # 确保我们得到的是1D数组
    #         if hasattr(similarity_scores, 'shape') and len(similarity_scores.shape) > 1:
    #             similarities = similarity_scores.flatten()
    #         else:
    #             similarities = similarity_scores
    #
    #         # 安全地获取最相似的记录
    #         if len(similarities) == 0:
    #             return []
    #
    #         # 使用安全的数组操作
    #         valid_indices = []
    #         for idx, score in enumerate(similarities):
    #             # 安全地检查相似度阈值
    #             if self.safe_array_check(score, 0.1):
    #                 valid_indices.append((idx, score))
    #
    #         # 按相似度排序
    #         valid_indices.sort(key=lambda x: x[1], reverse=True)
    #         similar_cases = []
    #
    #         for idx, score in valid_indices[:top_k]:
    #             try:
    #                 record = self.records[idx]
    #                 similar_cases.append({
    #                     'record': record,
    #                     'similarity': float(score),
    #                     'thing': record.thing,
    #                     'hexagram_name': record.hexagram_name,
    #                     'created_time': record.created_time
    #                 })
    #             except (IndexError, AttributeError) as e:
    #                 safe_print("[ERROR] 处理相似案例时出错: " + str(e))
    #                 continue
    #
    #         return similar_cases
    #
    #     except Exception as e:
    #         safe_print("[ERROR] 相似案例查找失败: " + str(e))
    #         return []

    def find_similar_cases(self, query, top_k=5):
        """查找相似的历史案例（简化修复版）"""
        if (self.similarity_matrix is None or
                not hasattr(self, 'records') or
                not self.records):
            return []

        try:
            # 预处理查询文本
            processed_query = self.preprocess_text(query)
            query_vector = self.vectorizer.transform([processed_query])

            # 计算相似度
            similarity_scores = cosine_similarity(query_vector, self.similarity_matrix)

            # 统一转换为1D numpy数组
            similarities = np.array(similarity_scores).flatten()

            # 获取top_k个最相似的索引
            if len(similarities) == 0:
                safe_print("[ERROR] 没有相似案例")
                return []

            # 使用argsort获取排序后的索引
            top_indices = np.argsort(similarities)[::-1][:top_k]

            similar_cases = []
            for idx in top_indices:
                if idx < len(self.records):
                    try:
                        record = self.records[idx]
                        score = similarities[idx]

                        similar_cases.append({
                            'record': record,
                            'similarity': float(score),
                            'thing': record.thing,
                            'hexagram_name': record.hexagram_name,
                            'created_time': record.created_time
                        })
                    except (IndexError, AttributeError) as e:
                        safe_print(f"[ERROR] 处理案例 {idx} 时出错: {e}")
                        continue

            return similar_cases

        except Exception as e:
            safe_print("[ERROR] 相似案例查找失败: " + str(e))
            return []



    def preprocess_text(self, text):
        """安全的文本预处理"""
        if not text:
            return ""

        try:
            # 简单的文本清理，避免复杂分词
            text = str(text)
            # 移除特殊字符但保留中文
            text = re.sub(r'[^\w\u4e00-\u9fff\s]', '', text)
            return text.strip()
        except Exception as e:
            safe_print("[ERROR] 文本预处理失败: " + str(e))
            return str(text)[:200]  # 返回截断的原始文本

    def generate_hexagram_interpretation(self, hexagram_name, user_thing, sentiment_analysis, changing_lines=None):
        """生成个性化的卦象解释 - 使用Ollama大模型"""
        # 优先使用Ollama大模型
        if OLLAMA_AVAILABLE:
            try:
                safe_print("正在使用Ollama大模型生成解释...")
                interpretation = ollama_client.generate_interpretation(
                    hexagram_name,
                    user_thing,
                    sentiment_analysis,
                    changing_lines
                )
                safe_print("✓ Ollama大模型解释生成成功")
                return interpretation
            except Exception as e:
                safe_print(f"Ollama大模型生成失败: {e}，使用备用解释")
                # 失败时回退到规则基础解释
                return self.get_fallback_interpretation(hexagram_name, user_thing, sentiment_analysis, changing_lines)
        else:
            # Ollama不可用时使用规则基础解释
            safe_print("使用规则基础解释")
            return self.get_fallback_interpretation(hexagram_name, user_thing, sentiment_analysis, changing_lines)

    def get_fallback_interpretation(self, hexagram_name, user_thing, sentiment_analysis):
        """备用解释生成（基于规则）"""
        sentiment = sentiment_analysis.get('sentiment', '中性')
        confidence = sentiment_analysis.get('confidence', 0.5)

        # 基于卦象和情感的规则解释
        interpretations = {
            '乾为天': {
                '积极': f"[SUCCESS] 乾卦象征天行健，君子以自强不息。关于'{user_thing}'，卦象显示您正处在积极向上的阶段，宜把握时机，勇往直前。",
                '消极': f"[THINK] 乾卦虽强，但亢龙有悔。对于'{user_thing}'，卦象提醒您需注意刚愎自用，适当调整策略，以柔克刚。",
                '中性': f"[BALANCE] 乾卦代表创造与领导。关于'{user_thing}'，卦象显示局势尚不明朗，建议您保持耐心，等待更好的时机。"
            },
            '坤为地': {
                '积极': f"[GROW] 坤卦象征地势坤，君子以厚德载物。关于'{user_thing}'，卦象显示宜稳扎稳打，积累实力，终有所成。",
                '消极': f"[IDEA] 坤卦主静，对于'{user_thing}'的挑战，卦象建议您以守为攻，积蓄力量，不宜贸然行动。",
                '中性': f"[PROCESS] 坤卦代表包容与承载。关于'{user_thing}'，卦象显示需要更多耐心，顺其自然会有转机。"
            },
            '水雷屯': {
                '积极': f"[GROW] 屯卦象征万物初生。关于'{user_thing}'，卦象显示虽然起步艰难，但只要坚持努力，必能开创局面。",
                '消极': f"[PROTECT] 屯卦提示初创之难。对于'{user_thing}'，卦象建议您谨慎行事，打好基础，避免急于求成。",
                '中性': f"[PROCESS] 屯卦代表起始与积累。关于'{user_thing}'，卦象显示需要循序渐进，不可操之过急。"
            },
            '巽为风': {
                '积极': f"[IDEA] 蒙卦象征启蒙与发展。关于'{user_thing}'，卦象显示需要学习和探索，将会获得新的认知和机会。",
                '消极': f"[THINK] 蒙卦提示迷茫之象。对于'{user_thing}'，卦象建议您寻求指导，多问多学，避免盲目行动。",
                '中性': f"[SEARCH] 蒙卦代表求知与启发。关于'{user_thing}'，卦象显示需要更多信息和思考才能做出决定。"
            },
            # 默认解释模板
            'default': {
                '积极': f"[TARGET] 关于'{user_thing}'，{hexagram_name}卦象显示积极态势。保持当前方向，注重实际行动。",
                '消极': f"[PROTECT] 面对'{user_thing}'的挑战，{hexagram_name}卦象建议谨慎行事，多听取他人意见。",
                '中性': f"[SEARCH] 关于'{user_thing}'，{hexagram_name}卦象显示需要更多观察。保持开放心态，等待时机成熟。"
            }
        }

        # 获取特定卦象的解释或使用默认解释
        hexagram_interpretations = interpretations.get(hexagram_name, interpretations['default'])
        interpretation = hexagram_interpretations.get(sentiment, interpretations['default']['中性'])

        # 添加置信度说明
        if confidence > 0.7:
            interpretation += " 卦象清晰，可信度较高。"
        elif confidence > 0.4:
            interpretation += " 卦象尚可，建议结合实际情况判断。"
        else:
            interpretation += " 卦象较为隐晦，仅供参考。"

        return interpretation


# 全局AI分析器实例
ai_analyzer = FortuneAIAnalyzer()
