from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from django.core.paginator import Paginator
from django.db.models import Q

from .deep_learning import ai_analyzer
from .models import FortuneRecord, HexagramInfo
from .fortune_logic import FortuneTeller
import json
import threading
import time
import random
from django.http import JsonResponse


def incense_burning(request):
    """焚香步骤 - 返回焚香进度"""
    if request.method == 'POST':
        # 从session中获取当前进度，如果没有则初始化为0
        progress = request.session.get('incense_progress', 0)

        steps = [
            "准备香炉...",
            "点燃香火...",
            "香烟袅袅升起...",
            "心中默念所求...",
            "香火渐旺...",
            "焚香完成..."
        ]
        step_index = request.session.get('step_index', 0)
        # 确保索引不超过步骤列表长度
        step_index = min(step_index, len(steps) - 1)
        step = steps[step_index]

        # 更新步骤索引
        if step_index < len(steps) - 1:
            request.session['step_index'] = step_index + 1
        else:
            # 如果已经是最后一步，重置索引
            request.session['step_index'] = 0
        # 更新进度
        if progress < 100:
            progress = min(progress + random.randint(10, 20), 100)
            # 将新进度保存到session
            request.session['incense_progress'] = progress

        # 如果进度完成，重置session中的进度
        if progress >= 100:
            request.session['incense_progress'] = 0

        # 返回焚香步骤
        return JsonResponse({
            'status': 'success',
            'step': step,
            'progress': progress
        })

    return JsonResponse({'status': 'error', 'message': '无效请求'})


def throwing_coins(request):
    """掷币步骤 - 返回掷币进度"""
    if request.method == 'POST':
        # 从session中获取当前进度，如果没有则初始化为0
        progress = request.session.get('coins_progress', 0)

        steps = [
            "手持三枚铜钱...",
            "静心凝神...",
            "记录爻象...",
            "第一次掷币...",
            "第二次掷币...",
            "第三次掷币...",
            "第四次掷币...",
            "第五次掷币...",
            "第六次掷币...",
            "卦象生成中..."
        ]
        step_index = request.session.get('step_index', 0)
        # 确保索引不超过步骤列表长度
        step_index = min(step_index, len(steps) - 1)
        step = steps[step_index]

        # 更新步骤索引
        if step_index < len(steps) - 1:
            request.session['step_index'] = step_index + 1
        else:
            # 如果已经是最后一步，重置索引
            request.session['step_index'] = 0
        # 更新进度
        if progress < 100:
            progress = min(progress + random.randint(8, 15), 100)
            # 将新进度保存到session
            request.session['coins_progress'] = progress

        # 如果进度完成，重置session中的进度
        if progress >= 100:
            request.session['coins_progress'] = 0

        # 返回掷币步骤
        return JsonResponse({
            'status': 'success',
            'step': step,
            'progress': progress
        })

    return JsonResponse({'status': 'error', 'message': '无效请求'})


# def calculate_with_animation(request):
#     """带动画的占卜计算"""
#     if request.method == 'POST':
#         thing = request.POST.get('thing', '').strip()
#         if not thing:
#             return render(request, 'zhouyi_app/index.html', {'error': '请输入占卜事项'})
#
#         # 将占卜事项存入session，以便在结果页面使用
#         request.session['fortune_thing'] = thing
#         request.session['fortune_time'] = timezone.now().isoformat()
#
#         return render(request, 'zhouyi_app/loading.html', {'thing': thing})
#
#     return redirect('fortune_app:index')

def calculate_with_animation(request):
    """带动画的占卜计算"""
    if request.method == 'POST':
        thing = request.POST.get('thing', '').strip()
        if not thing:
            return render(request, 'zhouyi_app/index.html', {'error': '请输入占卜事项'})

        # 将占卜事项存入session，以便在结果页面使用
        request.session['fortune_thing'] = thing
        request.session['fortune_time'] = timezone.now().isoformat()

        return render(request, 'zhouyi_app/loading.html', {'thing': thing})

    return redirect('fortune_app:index')


# def get_fortune_result(request):
#     """获取占卜结果（用于AJAX调用）"""
#     thing = request.session.get('fortune_thing', '未知事项')
#
#     # 创建算卦实例
#     teller = FortuneTeller()
#
#     # 生成卦象
#     hexagram_lines, hexagram_numbers, changing_lines = teller.generate_hexagram()
#
#     # 获取卦象信息
#     hexagram_info = teller.get_hexagram_info(hexagram_numbers)
#
#     # 获取客户端信息
#     ip_address = get_client_ip(request)
#     user_agent = request.META.get('HTTP_USER_AGENT', '')
#
#     # 保存到数据库
#     fortune_record = FortuneRecord(
#         thing=thing,
#         hexagram_lines='|'.join(hexagram_lines),
#         hexagram_numbers=''.join(str(num) for num in hexagram_numbers),
#         changing_lines=','.join(str(line) for line in changing_lines),
#         hexagram_name=hexagram_info['name'],
#         result_url=hexagram_info['url'],
#         ip_address=ip_address,
#         user_agent=user_agent
#     )
#     fortune_record.save()
#
#     # 返回结果
#     return JsonResponse({
#         'status': 'success',
#         'record_id': fortune_record.id,
#         'thing': thing,
#         'hexagram_lines': hexagram_lines,
#         'hexagram_numbers': hexagram_numbers,
#         'changing_lines': changing_lines,
#         'hexagram_name': hexagram_info['name'],
#         'result_url': hexagram_info['url'],
#         'hexagram_description': hexagram_info.get('description', '')
#     })

# def get_fortune_result(request):
#     """获取占卜结果（用于AJAX调用）"""
#     thing = request.session.get('fortune_thing', '未知事项')
#
#     # 创建算卦实例
#     teller = FortuneTeller()
#
#     # 生成卦象
#     hexagram_lines, hexagram_numbers, changing_lines = teller.generate_hexagram()
#
#     # 获取卦象信息
#     hexagram_info = teller.get_hexagram_info(hexagram_numbers)
#
#     # 获取客户端信息
#     ip_address = get_client_ip(request)
#     user_agent = request.META.get('HTTP_USER_AGENT', '')
#
#     # 保存到数据库
#     fortune_record = FortuneRecord(
#         thing=thing,
#         hexagram_lines='|'.join(hexagram_lines),
#         hexagram_numbers=''.join(str(num) for num in hexagram_numbers),
#         changing_lines=','.join(str(line) for line in changing_lines),
#         hexagram_name=hexagram_info['name'],
#         result_url=hexagram_info['url'],
#         ip_address=ip_address,
#         user_agent=user_agent
#     )
#     fortune_record.save()
#
#     # 在后台执行AI分析
#     threading.Thread(target=fortune_record.perform_ai_analysis).start()
#
#     # 返回结果
#     return JsonResponse({
#         'status': 'success',
#         'record_id': fortune_record.id,
#         'thing': thing,
#         'hexagram_lines': hexagram_lines,
#         'hexagram_numbers': hexagram_numbers,
#         'changing_lines': changing_lines,
#         'hexagram_name': hexagram_info['name'],
#         'result_url': hexagram_info['url'],
#         'hexagram_description': hexagram_info.get('description', ''),
#         'ai_analysis_pending': True  # 提示用户AI分析正在后台进行
#     })


# def index(request):
#     """首页"""
#     # 获取最近的几条记录
#     recent_records = FortuneRecord.objects.all().order_by('-created_time')[:5]
#     return render(request, 'zhouyi_app/index.html', {'recent_records': recent_records})

def index(request):
    """首页"""
    # 获取最近的几条记录
    recent_records = FortuneRecord.objects.all().order_by('-created_time')[:5]

    # 在后台训练相似度模型
    if recent_records:
        threading.Thread(
            target=ai_analyzer.train_similarity_model,
            args=(list(FortuneRecord.objects.all()[:100]),)
        ).start()

    return render(request, 'zhouyi_app/index.html', {'recent_records': recent_records})


def calculate_fortune(request):
    """计算卦象"""
    if request.method == 'POST':
        thing = request.POST.get('thing', '').strip()
        if not thing:
            return render(request, 'zhouyi_app/index.html', {'error': '请输入占卜事项'})

        # 创建算卦实例
        teller = FortuneTeller()

        # 生成卦象
        hexagram_lines, hexagram_numbers, changing_lines = teller.generate_hexagram()

        # 获取卦象信息
        hexagram_info = teller.get_hexagram_info(hexagram_numbers)

        # 获取客户端信息
        ip_address = get_client_ip(request)
        user_agent = request.META.get('HTTP_USER_AGENT', '')

        # 保存到数据库
        fortune_record = FortuneRecord(
            thing=thing,
            hexagram_lines='|'.join(hexagram_lines),
            hexagram_numbers=''.join(str(num) for num in hexagram_numbers),
            changing_lines=','.join(str(line) for line in changing_lines),
            hexagram_name=hexagram_info['name'],
            result_url=hexagram_info['url'],
            ip_address=ip_address,
            user_agent=user_agent
        )
        fortune_record.save()

        # 准备上下文数据
        context = {
            'thing': thing,
            'hexagram_lines': hexagram_lines,
            'hexagram_numbers': hexagram_numbers,
            'changing_lines': changing_lines,
            'hexagram_name': hexagram_info['name'],
            'hexagram_description': hexagram_info.get('description', ''),
            'result_url': hexagram_info['url'],
            'record_id': fortune_record.id,
            'created_time': timezone.now()
        }

        return render(request, 'zhouyi_app/result.html', context)

    return redirect('')


def fortune_history(request):
    """占卜历史"""
    records = FortuneRecord.objects.all().order_by('-created_time')

    # 分页
    paginator = Paginator(records, 20)  # 每页20条
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'zhouyi_app/history.html', {'page_obj': page_obj})


# def fortune_detail(request, record_id):
#     """占卜详情"""
#     record = get_object_or_404(FortuneRecord, id=record_id)
#     hexagram_lines = record.get_hexagram_lines_list()
#     changing_lines = record.get_changing_lines_list()
#
#     context = {
#         'record': record,
#         'hexagram_lines': hexagram_lines,
#         'changing_lines': changing_lines,
#     }
#     return render(request, 'zhouyi_app/detail.html', context)

def fortune_detail(request, record_id):
    """占卜详情"""
    record = get_object_or_404(FortuneRecord, id=record_id)
    hexagram_lines = record.get_hexagram_lines_list()
    changing_lines = record.get_changing_lines_list()

    # 如果AI分析尚未完成，尝试重新分析
    if not record.ai_interpretation:
        record.perform_ai_analysis()
        record.refresh_from_db()

    context = {
        'record': record,

        'hexagram_lines': hexagram_lines,
        'changing_lines': changing_lines,
        'has_ai_analysis': bool(record.ai_interpretation),
    }
    return render(request, 'zhouyi_app/detail.html', context)


def search_fortune(request):
    """搜索占卜记录"""
    query = request.GET.get('q', '').strip()
    if query:
        records = FortuneRecord.objects.filter(
            Q(thing__icontains=query) |
            Q(hexagram_name__icontains=query) |
            Q(hexagram_numbers__icontains=query)
        ).order_by('-created_time')
    else:
        records = FortuneRecord.objects.none()

    return render(request, 'zhouyi_app/search_results.html', {
        'records': records,
        'query': query
    })


def api_fortune(request):
    """API接口 - 返回JSON格式的占卜结果"""
    if request.method == 'POST':
        data = json.loads(request.body)
        thing = data.get('thing', '').strip()

        if not thing:
            return JsonResponse({'error': '请输入占卜事项'}, status=400)

        teller = FortuneTeller()
        hexagram_lines, hexagram_numbers, changing_lines = teller.generate_hexagram()
        hexagram_info = teller.get_hexagram_info(hexagram_numbers)

        # 保存记录
        fortune_record = FortuneRecord(
            thing=thing,
            hexagram_lines='|'.join(hexagram_lines),
            hexagram_numbers=''.join(str(num) for num in hexagram_numbers),
            changing_lines=','.join(str(line) for line in changing_lines),
            hexagram_name=hexagram_info['name'],
            result_url=hexagram_info['url'],
            ip_address=get_client_ip(request)
        )
        fortune_record.save()

        return JsonResponse({
            'success': True,
            'record_id': fortune_record.id,
            'thing': thing,
            'hexagram': {
                'name': hexagram_info['name'],
                'lines': hexagram_lines,
                'numbers': hexagram_numbers,
                'changing_lines': changing_lines,
                'description': hexagram_info.get('description', ''),
                'url': hexagram_info['url']
            },
            'created_time': timezone.now().isoformat()
        })

    return JsonResponse({'error': '只支持POST请求'}, status=405)


def get_client_ip(request):
    """获取客户端IP地址"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def get_ai_analysis(request, record_id):
    """获取AI分析结果（AJAX接口）"""
    record = get_object_or_404(FortuneRecord, id=record_id)

    # 如果AI分析尚未完成，执行分析
    if not record.ai_interpretation:
        record.perform_ai_analysis()
        record.refresh_from_db()

    return JsonResponse({
        'sentiment_analysis': record.sentiment_analysis or {},
        'ai_interpretation': record.ai_interpretation or '',
        'trend_prediction': record.trend_prediction or {},
        'similar_cases': record.similar_cases or [],
        'has_analysis': bool(record.ai_interpretation)
    })


def ai_insights(request):
    """AI洞察页面"""
    # 获取统计信息
    total_records = FortuneRecord.objects.count()
    recent_ai_analyzed = FortuneRecord.objects.filter(
        ai_interpretation__isnull=False
    ).order_by('-created_time')[:10]

    # 情感分布
    sentiment_data = {}
    for record in FortuneRecord.objects.filter(sentiment_analysis__isnull=False):
        sentiment = record.sentiment_analysis.get('sentiment', 'unknown')
        sentiment_data[sentiment] = sentiment_data.get(sentiment, 0) + 1

    context = {
        'total_records': total_records,
        'recent_ai_analyzed': recent_ai_analyzed,
        'sentiment_data': sentiment_data,
        'ai_enabled': ai_analyzer.sentiment_analyzer is not None,
    }

    return render(request, 'zhouyi_app/ai_insights.html', context)


def safe_ai_analysis(record):
    """安全的AI分析函数"""
    try:
        from .deep_learning import ai_analyzer

        # 情感分析
        sentiment_result = ai_analyzer.analyze_sentiment(record.thing)
        record.sentiment_analysis = sentiment_result

        # AI解释生成
        ai_interpretation = ai_analyzer.generate_hexagram_interpretation(
            record.hexagram_name,
            record.thing,
            sentiment_result
        )
        record.ai_interpretation = ai_interpretation

        # 趋势预测（基于历史数据）
        try:
            historical_data = FortuneRecord.objects.filter(
                hexagram_name=record.hexagram_name
            ).exclude(id=record.id)[:100]  # 限制数据量

            # 使用安全的统计方法
            total_cases = historical_data.count()
            if total_cases > 0:
                probability = min(total_cases / 1000, 0.3)  # 简化计算
                trend = "stable" if probability > 0.1 else "emerging"
                confidence = min(probability * 3, 0.8)
            else:
                probability = 0
                trend = "emerging"
                confidence = 0.6

            record.trend_prediction = {
                "trend": trend,
                "confidence": float(confidence),
                "probability": float(probability),
                "total_cases": total_cases
            }
        except Exception as e:
            print(f"趋势预测失败: {e}")
            record.trend_prediction = {"trend": "stable", "confidence": 0.5}

        # 相似案例查找
        try:
            similar_cases = ai_analyzer.find_similar_cases(record.thing, top_k=3)
            # 安全地处理相似案例数据
            safe_similar_cases = []
            for case in similar_cases:
                try:
                    safe_case = {
                        'id': getattr(case['record'], 'id', 0),
                        'thing': getattr(case['record'], 'thing', '未知'),
                        'hexagram_name': getattr(case['record'], 'hexagram_name', '未知'),
                        'similarity': float(case.get('similarity', 0)),
                        'time': getattr(case['record'], 'created_time', timezone.now()).isoformat()
                    }
                    safe_similar_cases.append(safe_case)
                except (AttributeError, KeyError) as e:
                    print(f"处理相似案例时出错: {e}")
                    continue

            record.similar_cases = safe_similar_cases
        except Exception as e:
            print(f"相似案例查找失败: {e}")
            record.similar_cases = []

        record.save()
        print(f"✅ AI分析完成: {record.thing}")

    except Exception as e:
        print(f"❌ AI分析失败: {e}")
        # 设置基本的AI分析字段
        record.sentiment_analysis = {"sentiment": "neutral", "confidence": 0.5}
        record.ai_interpretation = f"关于{record.thing}，{record.hexagram_name}卦象显示需要结合实际情况仔细分析。"
        record.trend_prediction = {"trend": "stable", "confidence": 0.5}
        record.similar_cases = []
        record.save()


def get_fortune_result(request):
    """获取占卜结果（修复版本）"""
    thing = request.session.get('fortune_thing', '未知事项')

    try:
        # 创建算卦实例
        from .fortune_logic import FortuneTeller
        teller = FortuneTeller()

        # 生成卦象
        hexagram_lines, hexagram_numbers, changing_lines = teller.generate_hexagram()

        # 获取卦象信息
        hexagram_info = teller.get_hexagram_info(hexagram_numbers)

        # 获取客户端信息
        ip_address = get_client_ip(request)
        user_agent = request.META.get('HTTP_USER_AGENT', '')[:500]  # 限制长度

        # 保存到数据库
        fortune_record = FortuneRecord(
            thing=thing,
            hexagram_lines='|'.join(hexagram_lines),
            hexagram_numbers=''.join(str(num) for num in hexagram_numbers),
            changing_lines=','.join(str(line) for line in changing_lines),
            hexagram_name=hexagram_info['name'],
            result_url=hexagram_info['url'],
            ip_address=ip_address,
            user_agent=user_agent
        )
        fortune_record.save()

        # 在后台执行AI分析（使用安全的函数）
        threading.Thread(target=safe_ai_analysis, args=(fortune_record,)).start()

        # 返回结果
        return JsonResponse({
            'status': 'success',
            'record_id': fortune_record.id,
            'thing': thing,
            'hexagram_lines': hexagram_lines,
            'hexagram_numbers': hexagram_numbers,
            'changing_lines': changing_lines,
            'hexagram_name': hexagram_info['name'],
            'result_url': hexagram_info['url'],
            'ai_analysis_pending': True
        })

    except Exception as e:
        print(f"❌ 获取卦象结果失败: {e}")
        return JsonResponse({
            'status': 'error',
            'message': '占卜过程出现错误，请稍后重试'
        }, status=500)
