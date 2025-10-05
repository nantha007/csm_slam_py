from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'csm_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nantha Kumar Sunder',
    maintainer_email='nantha07@terpmail.umd.edu',
    description='CSM SLAM',
    license='GNU General Public License v3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'csm_slam_node = csm_slam.csm_slam_online_node:main',
            'csm_slam_offline_node = csm_slam.csm_slam_offline_node:main'
        ],
    },
)
