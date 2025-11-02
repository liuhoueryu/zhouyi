#!/usr/bin/env python3
"""
修复Python环境的脚本
"""

import subprocess
import sys
import os


def run_command(command, description):
    """运行命令并显示结果"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}成功")
            return True
        else:
            print(f"❌ {description}失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}异常: {e}")
        return False


def main():
    print("🚀 开始修复Python环境...")

    # 1. 升级pip
    run_command("python -m pip install --upgrade pip", "升级pip")

    # 2. 修复NumPy兼容性
    print("🔄 修复NumPy兼容性...")
    commands = [
        "pip uninstall numpy -y",
        "pip cache purge",
        "pip install numpy==1.21.6",
    ]

    for cmd in commands:
        run_command(cmd, f"执行: {cmd}")

    # 3. 安装兼容的SciPy版本
    run_command("pip install scipy==1.7.3", "安装SciPy")

    # 4. 重新安装其他依赖
    packages = [
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "jieba==0.42.1",
    ]

    for package in packages:
        run_command(f"pip install {package}", f"安装{package}")

    # 5. 选择性安装AI库
    print("🤖 安装AI库（可选）...")
    ai_packages = [
        "torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu",
        "transformers==4.21.0",
    ]

    for package in ai_packages:
        run_command(f"pip install {package}", f"安装AI库")

    print("\n📊 环境修复完成！")
    print("💡 建议：如果仍有问题，考虑使用conda环境：")
    print("   conda create -n zhouyi python=3.8")
    print("   conda activate zhouyi")
    print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()