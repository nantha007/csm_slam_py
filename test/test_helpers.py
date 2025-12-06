# Copyright (C) 2025  Nantha Kumar Sunder
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Helper utilities for test files.

This module provides helper functions for loading settings and configuring
the test environment, such as adding virtual environment paths to sys.path.

Author: Nantha Kumar Sunder
"""

import os
import sys
import yaml


def load_settings():
    """
    Load settings from config/venv_settings.yaml file.

    This function loads the virtual environment path from the settings file
    and adds it to sys.path so that packages installed in the venv (like gtsam)
    can be imported in tests.

    Raises
    ------
    FileNotFoundError
        If the settings file is not found.

    """
    # Get the directory containing this file (test/)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the package root directory (parent of test/)
    package_root = os.path.dirname(test_dir)
    # Construct path to config directory
    config_dir = os.path.join(package_root, 'config')
    settings_file = os.path.join(config_dir, 'venv_settings.yaml')

    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f)
            venv_path = settings.get('venv_path', '')
            if venv_path:
                venv_path = os.path.expanduser(venv_path)
                if os.path.exists(venv_path):
                    if venv_path not in sys.path:
                        sys.path.append(venv_path)
                else:
                    # Warn but don't fail if venv path doesn't exist
                    print(f'Warning: Virtual environment path does not exist: {venv_path}')
    else:
        raise FileNotFoundError(f'Settings file not found: {settings_file}')

