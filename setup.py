from setuptools import setup, find_packages

setup(
    name="dacapo",
    version="0.1",
    url="https://github.com/funkelab/dacapo",
    author="Jan Funke",
    author_email="funkej@janelia.hhmi.org",
    license="MIT",
    packages=find_packages(),
    entry_points="""
            [console_scripts]
            dacapo=dacapo.cli.cli:cli
        """,
    include_package_data=True,
)
