from setuptools import setup

setup(
    entry_points={
        'console_scripts': [
            'tlsprint = tlsprint.cli:main',
        ],
    },
)
