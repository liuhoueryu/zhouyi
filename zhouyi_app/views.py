from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from django.core.paginator import Paginator
from django.db.models import Q
from .models import FortuneRecord, HexagramInfo
from .fortune_logic import FortuneTeller
import json

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


def get_fortune_result(request):
    """获取占卜结果（用于AJAX调用）"""
    thing = request.session.get('fortune_thing', '未知事项')

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
        'hexagram_description': hexagram_info.get('description', '')
    })


def index(request):
    """首页"""
    # 获取最近的几条记录
    recent_records = FortuneRecord.objects.all().order_by('-created_time')[:5]
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


def fortune_detail(request, record_id):
    """占卜详情"""
    record = get_object_or_404(FortuneRecord, id=record_id)
    hexagram_lines = record.get_hexagram_lines_list()
    changing_lines = record.get_changing_lines_list()

    context = {
        'record': record,
        'hexagram_lines': hexagram_lines,
        'changing_lines': changing_lines,
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
