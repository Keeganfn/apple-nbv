from setuptools import find_packages, setup

import os
import glob

package_name = 'nbv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Launch files
        # (os.path.join('share', package_name, 'launch'), glob.glob('launch/*launch.py')),

        # Config files
        # (os.path.join("share", package_name, "config"), glob.glob("config/*.yaml"))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lukestroh',
    maintainer_email='luke.strohbehn@gmail.com',
    description='rob599_hw2 package',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "apple_identifier = nbv.apple_identifier:main"    
        ],
    },
)