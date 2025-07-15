# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from aiq.data_models.component import AIQComponentEnum
from aiq.data_models.discovery_metadata import DiscoveryMetadata
from aiq.registry_handlers.package_utils import DependencyResolver
from aiq.registry_handlers.package_utils import build_aiq_artifact
from aiq.registry_handlers.package_utils import build_package_metadata
from aiq.registry_handlers.package_utils import build_wheel
from aiq.registry_handlers.schemas.package import WheelData
from aiq.registry_handlers.schemas.publish import AIQArtifact


class TestDependencyResolver:
    """Test cases for DependencyResolver class."""

    def setup_method(self):
        """Setup test environment."""
        self.resolver = DependencyResolver()

    def test_init_loads_metadata(self):
        """Test that __init__ properly loads metadata."""
        with patch('importlib.metadata.distributions') as mock_distributions:
            mock_dist = Mock()
            mock_dist.metadata.get.return_value = 'test-package'
            mock_dist.version = '1.0.0'
            mock_dist.requires = ['dependency1>=1.0', 'dependency2']
            mock_distributions.return_value = [mock_dist]

            resolver = DependencyResolver()

            assert 'test-package' in resolver._metadata_cache
            pkg_info = resolver._metadata_cache['test-package']
            assert pkg_info.get('version') == '1.0.0'
            assert pkg_info.get('requires') == ['dependency1>=1.0', 'dependency2']

    def test_normalize_name(self):
        """Test package name normalization."""
        test_cases = [
            ('package-name', 'package-name'),
            ('package_name', 'package-name'),
            ('Package_Name', 'package-name'),
            ('PACKAGE--NAME', 'package-name'),
            ('package___name', 'package-name'),
        ]

        for input_name, expected in test_cases:
            result = self.resolver._normalize_name(input_name)
            assert result == expected, f"Expected {expected}, got {result} for input {input_name}"

    def test_parse_requirement(self):
        """Test requirement string parsing."""
        test_cases = [
            ('requests>=2.0', 'requests'),
            ('Django==3.2.0', 'django'),
            ('package_name~=1.0', 'package-name'),
            ('invalid-package[extra]>=1.0', 'invalid-package'),
            ('package; python_version>="3.8"', 'package'),
            ('', None),
            ('invalid>>>package', 'invalid'),
        ]

        for req_str, expected in test_cases:
            result = self.resolver._parse_requirement(req_str)
            assert result == expected, f"Expected {expected}, got {result} for requirement {req_str}"

    def test_get_package_info(self):
        """Test getting package information."""
        # Mock some test data
        self.resolver._metadata_cache = {
            'test-package': {
                'name': 'test-package', 'version': '1.0.0', 'requires': ['dependency1']
            }
        }

        # Test existing package
        result = self.resolver.get_package_info('test-package')
        assert result is not None
        assert result['name'] == 'test-package'
        assert result['version'] == '1.0.0'

        # Test non-existent package
        result = self.resolver.get_package_info('non-existent-package')
        assert result is None

        # Test name normalization
        result = self.resolver.get_package_info('test_package')
        assert result is not None
        assert result['name'] == 'test-package'

    def test_get_all_dependencies(self):
        """Test dependency resolution."""
        # Mock metadata cache with dependency chain
        self.resolver._metadata_cache = {
            'root-package': {
                'name': 'root-package', 'version': '1.0.0', 'requires': ['dependency1>=1.0', 'dependency2']
            },
            'dependency1': {
                'name': 'dependency1', 'version': '1.0.0', 'requires': ['subdependency1']
            },
            'dependency2': {
                'name': 'dependency2', 'version': '2.0.0', 'requires': []
            },
            'subdependency1': {
                'name': 'subdependency1', 'version': '0.5.0', 'requires': []
            }
        }

        result = self.resolver.get_all_dependencies(('root-package', ))
        expected = frozenset({'root-package', 'dependency1', 'dependency2', 'subdependency1'})
        assert result == expected

    def test_get_all_dependencies_circular(self):
        """Test handling of circular dependencies."""
        # Mock circular dependency
        self.resolver._metadata_cache = {
            'package-a': {
                'name': 'package-a', 'version': '1.0.0', 'requires': ['package-b']
            },
            'package-b': {
                'name': 'package-b', 'version': '1.0.0', 'requires': ['package-a']
            }
        }

        result = self.resolver.get_all_dependencies(('package-a', ))
        expected = frozenset({'package-a', 'package-b'})
        assert result == expected

    def test_get_all_dependencies_missing_package(self):
        """Test handling of missing packages in dependency resolution."""
        self.resolver._metadata_cache = {
            'existing-package': {
                'name': 'existing-package', 'version': '1.0.0', 'requires': ['missing-package']
            }
        }

        result = self.resolver.get_all_dependencies(('existing-package', ))
        # Should include the existing package but skip missing dependency
        assert 'existing-package' in result

    def test_resolve_pyproject_dependencies(self):
        """Test resolving dependencies from pyproject.toml."""
        # Create a temporary pyproject.toml
        pyproject_content = """
[project]
dependencies = [
    "requests>=2.0",
    "click"
]

[project.optional-dependencies]
dev = ["pytest", "black"]
docs = ["sphinx"]

[dependency-groups]
test = ["coverage"]
lint = ["flake8"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(pyproject_content)
            temp_path = f.name

        try:
            # Mock metadata cache
            self.resolver._metadata_cache = {
                'requests': {
                    'name': 'requests', 'version': '2.25.1', 'requires': []
                },
                'click': {
                    'name': 'click', 'version': '8.0.0', 'requires': []
                },
                'pytest': {
                    'name': 'pytest', 'version': '6.0.0', 'requires': []
                },
                'black': {
                    'name': 'black', 'version': '21.0.0', 'requires': []
                },
                'sphinx': {
                    'name': 'sphinx', 'version': '4.0.0', 'requires': []
                },
                'coverage': {
                    'name': 'coverage', 'version': '5.0.0', 'requires': []
                },
                'flake8': {
                    'name': 'flake8', 'version': '3.8.0', 'requires': []
                },
            }

            # Test basic dependencies only
            result = self.resolver.resolve_pyproject_dependencies(temp_path)
            assert 'requests' in result
            assert 'click' in result
            assert 'pytest' not in result  # Should not include optional deps by default

            # Test with optional dependencies
            result = self.resolver.resolve_pyproject_dependencies(temp_path, include_optional_dependencies=True)
            assert 'requests' in result
            assert 'pytest' in result
            assert 'sphinx' in result
            assert 'coverage' not in result  # Should not include dependency groups

            # Test with dependency groups
            result = self.resolver.resolve_pyproject_dependencies(temp_path, include_dependency_groups=True)
            assert 'requests' in result
            assert 'coverage' in result
            assert 'flake8' in result
            assert 'pytest' not in result  # Should not include optional deps

            # Test with all dependencies
            result = self.resolver.resolve_pyproject_dependencies(temp_path,
                                                                  include_optional_dependencies=True,
                                                                  include_dependency_groups=True)
            assert len(result) == 7  # All packages should be included

        finally:
            os.unlink(temp_path)

    def test_resolve_pyproject_dependencies_file_not_found(self):
        """Test handling of missing pyproject.toml file."""
        with pytest.raises(FileNotFoundError):
            self.resolver.resolve_pyproject_dependencies('nonexistent.toml')

    def test_resolve_pyproject_dependencies_invalid_toml(self):
        """Test handling of invalid pyproject.toml file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write('invalid toml content [[[')
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                self.resolver.resolve_pyproject_dependencies(temp_path)
        finally:
            os.unlink(temp_path)

    def test_resolve_all_dependencies(self):
        """Test resolving all dependencies including optional ones."""
        pyproject_content = """
[project]
dependencies = ["requests"]

[project.optional-dependencies]
dev = ["pytest"]

[dependency-groups]
test = ["coverage"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(pyproject_content)
            temp_path = f.name

        try:
            # Mock metadata cache
            self.resolver._metadata_cache = {
                'requests': {
                    'name': 'requests', 'version': '2.25.1', 'requires': []
                },
                'pytest': {
                    'name': 'pytest', 'version': '6.0.0', 'requires': []
                },
                'coverage': {
                    'name': 'coverage', 'version': '5.0.0', 'requires': []
                },
            }

            result = self.resolver.resolve_all_dependencies(temp_path)

            # Should include all types of dependencies
            assert 'requests' in result
            assert 'pytest' in result
            assert 'coverage' in result

        finally:
            os.unlink(temp_path)

    def test_lru_cache_behavior(self):
        """Test that LRU cache is working for cached methods."""
        # Test _normalize_name caching
        result1 = self.resolver._normalize_name('Test_Package')
        result2 = self.resolver._normalize_name('Test_Package')
        assert result1 == result2 == 'test-package'

        # Check cache info (should have 1 hit for second call)
        cache_info = self.resolver._normalize_name.cache_info()
        assert cache_info.hits >= 1
        assert cache_info.misses >= 1

        # Test _parse_requirement caching
        result1 = self.resolver._parse_requirement('requests>=2.0')
        result2 = self.resolver._parse_requirement('requests>=2.0')
        assert result1 == result2 == 'requests'

        # Check cache info
        cache_info = self.resolver._parse_requirement.cache_info()
        assert cache_info.hits >= 1
        assert cache_info.misses >= 1

        # Test get_all_dependencies caching
        self.resolver._metadata_cache = {'test-pkg': {'name': 'test-pkg', 'version': '1.0.0', 'requires': []}}

        result1 = self.resolver.get_all_dependencies(('test-pkg', ))
        result2 = self.resolver.get_all_dependencies(('test-pkg', ))
        assert result1 == result2 == frozenset({'test-pkg'})

        # Check cache info
        cache_info = self.resolver.get_all_dependencies.cache_info()
        assert cache_info.hits >= 1
        assert cache_info.misses >= 1


def test_build_wheel():

    package_root = "."

    wheel_data = build_wheel(package_root=package_root)

    assert isinstance(wheel_data, WheelData)
    assert wheel_data.package_root == package_root


@pytest.mark.parametrize("use_wheel_data", [
    (True),
    (False),
])
def test_build_package_metadata(use_wheel_data):

    wheel_data: WheelData | None = None
    if (use_wheel_data):
        wheel_data = WheelData(package_root=".",
                               package_name="aiq",
                               toml_project={},
                               union_dependencies=set(),
                               whl_path="whl/path.whl",
                               whl_base64="",
                               whl_version="")

    discovery_metadata = build_package_metadata(wheel_data=wheel_data)

    assert isinstance(discovery_metadata, dict)

    for component_type, discovery_metadatas in discovery_metadata.items():
        assert isinstance(component_type, AIQComponentEnum)

        for discovery_metadata_item in discovery_metadatas:
            assert isinstance(discovery_metadata_item, dict)
            # Verify it can be converted to DiscoveryMetadata
            DiscoveryMetadata(**discovery_metadata_item)


def test_build_aiq_artifact():

    package_root = "."

    aiq_artifact = build_aiq_artifact(package_root=package_root)

    assert isinstance(aiq_artifact, AIQArtifact)
