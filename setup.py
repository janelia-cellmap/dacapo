from setuptools import setup

setup(
    name="dacapo",
    version="0.1",
    url="https://github.com/funkelab/dacapo",
    author="Jan Funke, Will Patton",
    author_email="funkej@janelia.hhmi.org, pattonw@janelia.hhmi.org",
    license="MIT",
    packages=[
        'dacapo',
        'dacapo.experiments',
        'dacapo.experiments.architectures',
        'dacapo.experiments.datasets',
        'dacapo.experiments.tasks',
        'dacapo.experiments.tasks.losses',
        'dacapo.experiments.tasks.post_processors',
        'dacapo.experiments.tasks.predictors',
        'dacapo.experiments.trainers'
    ],
    entry_points={
        'console_scripts': [
            'dacapo=scripts.dacapo:cli'
        ]
    },
    include_package_data=True
)
