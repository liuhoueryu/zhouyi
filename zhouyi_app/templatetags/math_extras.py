# zhouyi_app/templatetags/math_extras.py
from django import template

register = template.Library()

@register.filter
def mul(value, arg):
    """乘法过滤器：将value乘以arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0
