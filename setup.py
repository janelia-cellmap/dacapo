from setuptools import setup

setup(
        name='dacapo',
        version='0.1',
        url='https://github.com/funkelab/dacapo',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'dacapo',
            'dacapo.evaluate',
            'dacapo.models'
        ]
)
