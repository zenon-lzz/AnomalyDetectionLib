"""
=================================================
@Author: Zenon
@Date: 2025-05-01
@Description：Python Grammar Learning
==================================================
"""
import unittest


class GrammarLearning(unittest.TestCase):

    def test_basic(self):
        from datetime import datetime

        # 获取当前日期和时间
        now = datetime.now()

        # 格式化输出
        formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"当前日期时间: {formatted_date}")
        print(list(range(10, 210, 10)))
