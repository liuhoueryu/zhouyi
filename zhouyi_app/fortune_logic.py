import random
import time
from django.utils import timezone
from .models import HexagramInfo


class FortuneTeller:
    def __init__(self):
        self.gua_dic = self._init_gua_dictionary()
        self._load_hexagram_from_db()

    def _init_gua_dictionary(self):
        """初始化默认卦象字典"""
        return {
            '111111': {'name': '乾为天', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-1.html'},
            '000000': {'name': '坤为地', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-2.html'},
            '010001': {'name': '水雷屯', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-3.html'},
            '100010': {'name': '山水蒙', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-4.html'},
            '010111': {'name': '水天需', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-5.html'},
            '111010': {'name': '天水讼', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-6.html'},
            '000010': {'name': '地水师', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-7.html'},
            '010000': {'name': '水地比', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-8.html'},
            '110111': {'name': '风天小畜', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-9.html'},
            '111011': {'name': '天泽履', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-10.html'},
            '000111': {'name': '地天泰', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-11.html'},
            '111000': {'name': '天地否', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-12.html'},
            '111101': {'name': '天火同人', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-13.html'},
            '101111': {'name': '火天大有', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-14.html'},
            '000100': {'name': '地山谦', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-15.html'},
            '001000': {'name': '雷地豫', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-16.html'},
            '011001': {'name': '泽雷随', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-17.html'},
            '100110': {'name': '山风蛊', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-18.html'},
            '000011': {'name': '地泽临', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-19.html'},
            '110000': {'name': '风地观', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-20.html'},
            '101001': {'name': '火雷噬嗑', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-21.html'},
            '100101': {'name': '山火贲', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-22.html'},
            '100000': {'name': '山地剥', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-23.html'},
            '000001': {'name': '地雷复', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-24.html'},
            '111001': {'name': '天雷无妄', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-25.html'},
            '100111': {'name': '山天大畜', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-26.html'},
            # 注：与22重复，实际应为“山火贲”
            '100001': {'name': '山雷颐', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-27.html'},
            '011110': {'name': '泽风大过', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-28.html'},
            '010010': {'name': '坎为水', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-29.html'},
            # 注：与27重复，实际应为“山火贲”仅一卦
            '101101': {'name': '离为火', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-30.html'},
            '011100': {'name': '泽山咸', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-31.html'},
            '001110': {'name': '雷风恒', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-32.html'},
            '111100': {'name': '天山遯', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-33.html'},
            # 注：与第9卦重复，应为“雷天大壮”
            '001111': {'name': '雷天大壮', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-34.html'},
            '101000': {'name': '火地晋', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-35.html'},  # 注：与第16卦重复
            '000101': {'name': '地火明夷', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-36.html'},
            '110101': {'name': '风火家人', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-37.html'},
            '101011': {'name': '火泽睽', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-38.html'},
            '010100': {'name': '水山蹇', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-39.html'},
            '001010': {'name': '雷水解', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-40.html'},
            '100011': {'name': '山泽损', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-41.html'},
            # 注：与38重复，应为“泽水困”或“天雷无妄”
            '110001': {'name': '风雷益', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-42.html'},
            '011111': {'name': '泽天夬', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-43.html'},
            '111110': {'name': '天风姤', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-44.html'},  # 注：与42重复
            '011000': {'name': '泽地萃', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-45.html'},
            '000110': {'name': '地风升', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-46.html'},
            '011010': {'name': '泽水困', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-47.html'},
            '010110': {'name': '水风井', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-48.html'},
            '011101': {'name': '泽火革', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-49.html'},  # 与21重复
            '101110': {'name': '火风鼎', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-50.html'},  # 与40重复
            '001001': {'name': '震为雷', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-51.html'},  # 与25重复
            '100100': {'name': '艮为山', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-52.html'},
            '110100': {'name': '风山渐', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-53.html'},
            '001011': {'name': '雷泽归妹', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-54.html'},
            '001101': {'name': '雷火丰', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-55.html'},
            '101100': {'name': '火山旅', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-56.html'},  # 与46重复
            '110110': {'name': '巽为风', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-57.html'},
            '011011': {'name': '兑为泽', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-58.html'},
            '110010': {'name': '风水涣', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-59.html'},
            '010011': {'name': '水泽节', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-60.html'},
            '110011': {'name': '风泽中孚', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-61.html'},
            '001100': {'name': '雷山小过', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-62.html'},
            '010101': {'name': '水火既济', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-63.html'},
            '101010': {'name': '火水未济', 'url': 'https://www.buyiju.com/zhouyi/yijing/64gua-64.html'}
        }

    def _load_hexagram_from_db(self):
        """从数据库加载卦象信息"""
        try:
            hexagrams = HexagramInfo.objects.all()
            for hexagram in hexagrams:
                self.gua_dic[hexagram.number] = {
                    'name': hexagram.name,
                    'url': hexagram.url,
                    'description': hexagram.description
                }
        except Exception as e:
            # 如果数据库表不存在，暂时使用默认字典
            print(f"加载卦象信息失败: {e}")

    def throw_coin(self):
        """掷硬币得爻"""
        throws = [random.randint(0, 1) for _ in range(3)]
        total = sum(throws)

        if total == 0:  # 老阴
            return "- -   x", 0, True
        elif total == 1:  # 少阳
            return "—", 1, False
        elif total == 3:  # 老阳
            return "—   x", 1, True
        else:  # 少阴
            return "- -", 0, False

    def generate_hexagram(self):
        """生成卦象"""
        hexagram_lines = []
        hexagram_numbers = []
        changing_lines = []

        for i in range(6):
            line, number, is_changing = self.throw_coin()
            hexagram_lines.append(line)
            hexagram_numbers.append(number)
            if is_changing:
                changing_lines.append(i)

        # 卦象从下往上排列，但存储时反转
        hexagram_lines.reverse()
        hexagram_numbers.reverse()

        return hexagram_lines, hexagram_numbers, changing_lines

    def get_hexagram_info(self, hexagram_numbers):
        """获取卦象信息"""
        hexagram_key = ''.join(str(num) for num in hexagram_numbers)
        hexagram_info = self.gua_dic.get(hexagram_key, {'name': '未知卦象', 'url': '#'})

        return {
            'name': hexagram_info.get('name', '未知卦象'),
            'url': hexagram_info.get('url', '#'),
            'description': hexagram_info.get('description', '')
        }