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
        import pandas as pd

        # 示例：字典数组
        data = [
            {"Name": 1, "Age": 25, "City": 4},
            {"Name": 2, "Age": 30, "City": 5},
            {"Name": 3, "Age": 22, "City": 6}
        ]

        # 转换为 DataFrame 并打印
        df = pd.DataFrame(data)
        print(df)
        print(df.mean().round(2).to_string())
