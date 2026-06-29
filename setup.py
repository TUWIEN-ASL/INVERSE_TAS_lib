from setuptools import setup, find_packages

setup(
    name='INVERSE_TAS',  # Replace with your project name
    version='0.1.0',           # Replace with your project's version
    description='A pipeline for traning and deploying temporal action segmentation models',  # Add a description
    # long_description=open('README.md').read(),  # Optional, if you have a README file
    # long_description_content_type='text/markdown',  # Adjust if your README is not markdown
    author='Daniel Sliwowski',  # Replace with your name
    author_email='daniel.sliwowski@tuwien.ac.at',  # Replace with your email
    # url='https://github.com/yourusername/your_project',  # Replace with your project URL (if available)
    packages=find_packages(),  # Automatically finds all packages in your project
    # install_requires=[  # List any dependencies your project requires
    #     'dependency1',
    #     'dependency2',
    # ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Adjust according to your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Adjust the minimum Python version required
)