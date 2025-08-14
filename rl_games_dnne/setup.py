from setuptools import setup, find_packages

setup(
    name='rl_games_dnne',
    version='1.0.0',
    packages=find_packages(),
    description='DNNE version of rl_games with additional features',
    author='DNNE Team',
    python_requires='>=3.8',
    install_requires=[
        'rl_games>=1.6.0',
    ],
)