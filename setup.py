from setuptools import setup, find_packages

setup(
    name='tsadlib',
    version='0.0.1',
    packages=find_packages(include=['tsadlib', 'tsadlib.*']),  # Explicitly include all submodules
    package_data={
        'tsadlib': ['**/*.py'],  # Ensure all Python files are included
    },
    include_package_data=True,  # Enable inclusion of non-Python files
    url='https://github.com/zenon-lzz/AnomalyDetectionLib',
    license='BSD 3-Clause',
    author='Zenon Liu',
    author_email='2549562253@qq.com',
    description='A unified benchmark for anomaly detection models',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[  # Add required dependencies
        'torch>=1.8.0',
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
    ],
)
