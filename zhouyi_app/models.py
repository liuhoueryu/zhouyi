
# Create your models here.
from django.db import models
from django.utils import timezone
import json


class FortuneRecord(models.Model):
    thing = models.CharField(max_length=200, verbose_name="占卜事项")
    hexagram_lines = models.CharField(max_length=200, verbose_name="卦爻")
    hexagram_numbers = models.CharField(max_length=6, verbose_name="卦象数字")
    changing_lines = models.CharField(max_length=50, verbose_name="变爻位置", blank=True)
    hexagram_name = models.CharField(max_length=50, verbose_name="卦象名称", default="未知卦象")
    result_url = models.URLField(verbose_name="解卦链接")
    created_time = models.DateTimeField(default=timezone.now, verbose_name="创建时间")
    ip_address = models.GenericIPAddressField(verbose_name="IP地址", null=True, blank=True)
    user_agent = models.TextField(verbose_name="浏览器信息", blank=True)

    # AI分析字段
    sentiment_analysis = models.JSONField(verbose_name="情感分析", null=True, blank=True)
    ai_interpretation = models.TextField(verbose_name="AI解释", blank=True)
    trend_prediction = models.JSONField(verbose_name="趋势预测", null=True, blank=True)
    similar_cases = models.JSONField(verbose_name="相似案例", null=True, blank=True)

    class Meta:
        verbose_name = "占卜记录"
        verbose_name_plural = "占卜记录"
        ordering = ['-created_time']
        indexes = [
            models.Index(fields=['created_time']),
            models.Index(fields=['hexagram_numbers']),
            models.Index(fields=['hexagram_name']),
        ]

    def __str__(self):
        return f"{self.thing} - {self.hexagram_name} - {self.created_time.strftime('%Y-%m-%d %H:%M')}"

    def get_hexagram_lines_list(self):
        """将存储的卦爻字符串转换为列表"""
        return self.hexagram_lines.split('|')

    def get_changing_lines_list(self):
        """将存储的变爻位置字符串转换为列表"""
        if self.changing_lines:
            return [int(x) for x in self.changing_lines.split(',')]
        return []

    def to_dict(self):
        """将记录转换为字典，便于API调用"""
        return {
            'id': self.id,
            'thing': self.thing,
            'hexagram_name': self.hexagram_name,
            'hexagram_lines': self.get_hexagram_lines_list(),
            'hexagram_numbers': self.hexagram_numbers,
            'changing_lines': self.get_changing_lines_list(),
            'result_url': self.result_url,
            'created_time': self.created_time.isoformat(),
            'sentiment_analysis': self.sentiment_analysis,
            'ai_interpretation': self.ai_interpretation,
            'trend_prediction': self.trend_prediction,
        }

    def perform_ai_analysis(self):
        """执行AI分析"""
        from .deep_learning import ai_analyzer

        try:
            # 情感分析
            self.sentiment_analysis = ai_analyzer.analyze_sentiment(self.thing)

            # AI解释生成
            self.ai_interpretation = ai_analyzer.generate_hexagram_interpretation(
                self.hexagram_name,
                self.thing,
                self.sentiment_analysis
            )

            # 趋势预测（基于历史数据）
            historical_data = FortuneRecord.objects.filter(
                hexagram_name=self.hexagram_name
            ).exclude(id=self.id)

            # 使用新的predict_fortune_trend方法
            self.trend_prediction = ai_analyzer.predict_fortune_trend(
                self.hexagram_name,
                historical_data
            )

            # 相似案例查找
            similar_cases = ai_analyzer.find_similar_cases(self.thing, top_k=3)
            self.similar_cases = [
                {
                    'id': case['record'].id,
                    'thing': case['thing'],
                    'hexagram_name': case['hexagram_name'],
                    'similarity': case['similarity'],
                    'time': case['created_time'].isoformat()
                }
                for case in similar_cases
            ]

            self.save()
            return True

        except Exception as e:
            print(f"AI分析失败: {e}")
            # 设置基本的AI分析字段作为回退
            self.sentiment_analysis = {"sentiment": "neutral", "confidence": 0.5}
            self.ai_interpretation = f"关于{self.thing}，{self.hexagram_name}卦象显示需要结合实际情况仔细分析。"
            self.trend_prediction = {"trend": "stable", "confidence": 0.5}
            self.similar_cases = []
            self.save()
            return False

class HexagramInfo(models.Model):
    """卦象信息表"""
    number = models.CharField(max_length=6, unique=True, verbose_name="卦象数字")
    name = models.CharField(max_length=50, verbose_name="卦名")
    description = models.TextField(verbose_name="卦象描述", blank=True)
    url = models.URLField(verbose_name="详细解释链接")
    created_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "卦象信息"
        verbose_name_plural = "卦象信息"

    def __str__(self):
        return f"{self.name} ({self.number})"