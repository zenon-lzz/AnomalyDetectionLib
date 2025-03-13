from setuptools import setup, find_packages

setup(
    name='tsadlib',
    version='0.1.0',
    packages=find_packages(where="tsadlib"),
    package_dir={"": "tsadlib"},
    url='',
    license='BSD 3-Clause',
    author='liuzhenzhou',
    author_email='2549562253@qq.com',
    description='A unified benchmark for anomaly detection models'
)
