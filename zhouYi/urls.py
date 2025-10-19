"""zhouYi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from zhouyi_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('calculate/', views.calculate_fortune, name='calculate_fortune'),
    path('calculate-animation/', views.calculate_with_animation, name='calculate_with_animation'),
    path('get_fortune_result/', views.get_fortune_result, name='get_fortune_result'),
    path('incense_burning/', views.incense_burning, name='incense_burning'),
    path('throwing_coins/', views.throwing_coins, name='throwing_coins'),
    path('index/', views.index),
    path('history/', views.fortune_history, name='fortune_history'),
    path('detail/<int:record_id>/', views.fortune_detail, name='fortune_detail'),
]
