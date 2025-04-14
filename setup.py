from setuptools import setup, find_packages

setup(
    # 包名称 (必需)
    name='tsadlib',
    # 版本号 (必需)，遵循语义化版本规范
    version='0.0.1',

    # 自动查找包，排除tests目录
    packages=find_packages(exclude=['tests']),

    # 项目URL (推荐)
    url='https://github.com/zenon-lzz/AnomalyDetectionLib',

    # 许可证类型 (推荐)
    license='BSD 3-Clause',

    # 作者信息 (推荐)
    author='Zenon Liu',
    author_email='2549562253@qq.com',

    # 简短描述 (必需)
    description='A unified benchmark for anomaly detection models',

    # 详细描述 (推荐)
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    # 分类信息 (推荐)
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
