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

import base64
import importlib.metadata
import importlib.util
import logging
import os
import re
import subprocess
import tomllib
from functools import lru_cache

from packaging.requirements import Requirement
from pkginfo import Wheel

from aiq.data_models.component import AIQComponentEnum
from aiq.data_models.discovery_metadata import DiscoveryMetadata
from aiq.registry_handlers.schemas.package import WheelData
from aiq.registry_handlers.schemas.publish import AIQArtifact
from aiq.runtime.loader import PluginTypes
from aiq.runtime.loader import discover_entrypoints

# pylint: disable=redefined-outer-name
logger = logging.getLogger(__name__)


class DependencyResolver:
    """Offline dependency resolver using importlib.metadata."""

    def __init__(self):
        """Initialize resolver with local package metadata."""
        self._metadata_cache = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load all package metadata once for fast lookup."""
        unique_packages = set()

        for dist in importlib.metadata.distributions():
            name = dist.metadata.get('name', '').lower()
            if name:
                unique_packages.add(name)

                # Normalize name (replace underscores/hyphens)
                normalized = re.sub(r'[-_]+', '-', name)

                self._metadata_cache[normalized] = {
                    'name': name, 'version': dist.version, 'requires': dist.requires or []
                }

                # Also cache with original name and underscore variant
                self._metadata_cache[name] = self._metadata_cache[normalized]
                underscore_name = name.replace('-', '_')
                if underscore_name != name:
                    self._metadata_cache[underscore_name] = (self._metadata_cache[normalized])

    @lru_cache(maxsize=128)
    def _normalize_name(self, name: str) -> str:
        """Normalize package name for consistent lookup."""
        return re.sub(r'[-_]+', '-', name.lower())

    @lru_cache(maxsize=256)
    def _parse_requirement(self, req_str: str) -> str | None:
        """Parse requirement string and return package name."""
        try:
            req = Requirement(req_str)
            return self._normalize_name(req.name)
        except Exception:
            # Fallback to simple parsing
            name = re.split(r'[<>=~!;\[]', req_str)[0].strip()
            return self._normalize_name(name) if name else None

    @lru_cache(maxsize=64)
    def get_all_dependencies(self, root_packages: tuple[str, ...]) -> frozenset[str]:
        """
        Get all dependencies for given root packages using BFS.

        Args:
            root_packages (tuple[str, ...]): Tuple of root package names

        Returns:
            frozenset[str]: Set of all package names (including root packages)
        """
        all_packages = set()
        to_process = set(self._normalize_name(pkg) for pkg in root_packages)
        visited = set()

        while to_process:
            current = to_process.pop()

            if current in visited:
                continue

            visited.add(current)
            all_packages.add(current)

            # Get package metadata
            pkg_info = self._metadata_cache.get(current)
            if not pkg_info:
                continue

            # Add dependencies to process
            for req_str in pkg_info.get('requires', []):
                dep_name = self._parse_requirement(req_str)
                if dep_name and dep_name not in visited:
                    to_process.add(dep_name)

        return frozenset(all_packages)

    def get_package_info(self, package_name: str) -> dict | None:
        """Get package information if available locally.

        Args:
            package_name (str): Name of the package to get information for

        Returns:
            dict | None: Package information if available, None otherwise
        """
        normalized = self._normalize_name(package_name)
        return self._metadata_cache.get(normalized)

    def resolve_pyproject_dependencies(self,
                                       pyproject_path: str = "pyproject.toml",
                                       include_dependency_groups: bool = False,
                                       include_optional_dependencies: bool = False) -> dict[str, dict]:
        """
        Resolve all dependencies from pyproject.toml.

        Args:
            pyproject_path (str): Path to pyproject.toml file
            include_dependency_groups (bool): Whether to include dependency-groups
            include_optional_dependencies (bool): Whether to include project.optional-dependencies

        Returns:
            dict[str, dict]: Dict mapping package names to their info
        """
        # Parse pyproject.toml
        try:
            with open(pyproject_path, 'rb') as f:
                data = tomllib.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to parse pyproject.toml: {exc}") from exc

        # Extract dependencies
        dependencies = []

        # Main dependencies
        project_deps = data.get("project", {}).get("dependencies", [])
        dependencies.extend(project_deps)

        # Optional dependencies (project.optional-dependencies)
        if include_optional_dependencies:
            optional_deps = data.get("project", {}).get("optional-dependencies", {})
            for group_deps in optional_deps.values():
                dependencies.extend(group_deps)

        # Dependency groups (dependency-groups)
        if include_dependency_groups:
            dep_groups = data.get("dependency-groups", {})
            for group_deps in dep_groups.values():
                dependencies.extend(group_deps)

        # Parse root packages
        root_packages = []
        for dep in dependencies:
            pkg_name = self._parse_requirement(dep)
            if pkg_name:
                root_packages.append(pkg_name)

        # Get all dependencies
        all_packages = self.get_all_dependencies(tuple(root_packages))

        # Build result with package info
        result = {}
        locally_found = 0

        for pkg_name in all_packages:
            pkg_info = self.get_package_info(pkg_name)
            if pkg_info:
                result[pkg_name] = {'name': pkg_info['name'], 'version': pkg_info['version'], 'found_locally': True}
                locally_found += 1
            else:
                result[pkg_name] = {'name': pkg_name, 'version': None, 'found_locally': False}

        return result

    def resolve_all_dependencies(self, pyproject_path: str = "pyproject.toml") -> dict[str, dict]:
        """
        Resolve all dependencies from pyproject.toml including optional ones.

        Args:
            pyproject_path (str): Path to pyproject.toml file

        Returns:
            dict[str, dict]: Dict mapping package names to their info
        """
        return self.resolve_pyproject_dependencies(pyproject_path=pyproject_path,
                                                   include_dependency_groups=True,
                                                   include_optional_dependencies=True)


@lru_cache
def get_module_name_from_distribution(distro_name: str) -> str | None:
    """Return the first top-level module name for a given distribution name."""
    if not distro_name:
        return None

    try:
        # Read 'top_level.txt' which contains the module(s) provided by the package
        dist = importlib.metadata.distribution(distro_name)
        # will reading a file set of vun scan?
        top_level = dist.read_text('top_level.txt')

        if top_level:
            module_names = top_level.strip().split()
            # return firs module name
            return module_names[0]
    except importlib.metadata.PackageNotFoundError:
        # Distribution not found
        return None
    except FileNotFoundError:
        # 'top_level.txt' might be missing
        return None

    return None


def build_wheel(package_root: str) -> WheelData:
    """Builds a Python .whl for the specified package and saves to disk, sets self._whl_path, and returned as bytes.

    Args:
        package_root (str): Path to the local package repository.

    Returns:
        WheelData: Data model containing a built python wheel and its corresponding metadata.
    """

    pyproject_toml_path = os.path.join(package_root, "pyproject.toml")

    resolver = DependencyResolver()
    all_dependencies = resolver.resolve_all_dependencies(pyproject_path=pyproject_toml_path)
    union_dependencies = set(k for k, v in all_dependencies.items() if v["found_locally"])

    if not os.path.exists(pyproject_toml_path):
        raise ValueError("Invalid package path, does not contain a pyproject.toml file.")

    with open(pyproject_toml_path, "rb") as f:
        data = tomllib.load(f)

    toml_project: dict = data.get("project", {})
    toml_project_name = toml_project.get("name", None)

    assert toml_project_name is not None, f"Package name '{toml_project_name}' not found in pyproject.toml"
    module_name = get_module_name_from_distribution(toml_project_name)
    assert module_name is not None, f"No modules found for package name '{toml_project_name}'"

    assert importlib.util.find_spec(module_name) is not None, (f"Package {module_name} not "
                                                               "installed, cannot discover components.")

    union_dependencies.add(toml_project_name)

    working_dir = os.getcwd()
    os.chdir(package_root)

    # Ensure build happens in the correct directory and dist is created here
    result = subprocess.run(["uv", "build", "--wheel", "--out-dir", "dist"], check=True)
    result.check_returncode()

    # The dist directory should now be in the current directory (package root)
    if not os.path.exists("dist"):
        raise FileNotFoundError(f"Build failed: dist directory not found in package root {os.getcwd()}")

    whl_files = [f for f in os.listdir("dist") if f.endswith('.whl')]
    if not whl_files:
        raise FileNotFoundError(f"No wheel files found in {os.getcwd()}/dist")

    whl_file = sorted(whl_files, reverse=True)[0]
    whl_file_path = os.path.join("dist", whl_file)

    with open(whl_file_path, "rb") as whl:
        whl_bytes = whl.read()
        whl_base64 = base64.b64encode(whl_bytes).decode("utf-8")

    # Create absolute path for the wheel file
    whl_path = os.path.join(os.getcwd(), whl_file_path)

    os.chdir(working_dir)

    whl_version = Wheel(whl_path).version or "unknown"

    return WheelData(
        package_root=package_root,
        package_name=module_name,  # should it be module name or distro name here
        toml_project=toml_project,
        union_dependencies=union_dependencies,
        whl_path=whl_path,
        whl_base64=whl_base64,
        whl_version=whl_version)


def build_package_metadata(wheel_data: WheelData | None) -> dict[AIQComponentEnum, list[DiscoveryMetadata]]:
    """Loads discovery metadata for all registered AIQ Toolkit components included in this Python package.

    Args:
        wheel_data (WheelData): Data model containing a built python wheel and its corresponding metadata.

    Returns:
        dict[AIQComponentEnum, list[DiscoveryMetadata]]: List containing each components discovery
        metadata.
    """

    from aiq.cli.type_registry import GlobalTypeRegistry
    from aiq.registry_handlers.metadata_factory import ComponentDiscoveryMetadata
    from aiq.runtime.loader import discover_and_register_plugins

    discover_and_register_plugins(PluginTypes.ALL)

    registry = GlobalTypeRegistry.get()

    aiq_plugins = discover_entrypoints(PluginTypes.ALL)

    if (wheel_data is not None):
        registry.register_package(package_name=wheel_data.package_name, package_version=wheel_data.whl_version)
        for entry_point in aiq_plugins:
            package_name = entry_point.module.split('.')[0]
            if (package_name == wheel_data.package_name):
                continue
            if (package_name in wheel_data.union_dependencies):
                registry.register_package(package_name=package_name)

    else:
        for entry_point in aiq_plugins:
            package_name = entry_point.module.split('.')[0]
            registry.register_package(package_name=package_name)

    discovery_metadata = {}
    for component_type in AIQComponentEnum:

        if (component_type == AIQComponentEnum.UNDEFINED):
            continue
        component_metadata = ComponentDiscoveryMetadata.from_package_component_type(wheel_data=wheel_data,
                                                                                    component_type=component_type)
        component_metadata.load_metadata()
        discovery_metadata[component_type] = component_metadata.get_metadata_items()

    return discovery_metadata


def build_aiq_artifact(package_root: str) -> AIQArtifact:
    """Builds a complete AIQ Toolkit Artifact that can be published for discovery and reuse.

    Args:
        package_root (str): Path to root of python package

    Returns:
        AIQArtifact: An publishabla AIQArtifact containing package wheel and discovery metadata.
    """

    from aiq.registry_handlers.schemas.publish import BuiltAIQArtifact

    wheel_data = build_wheel(package_root=package_root)
    metadata = build_package_metadata(wheel_data=wheel_data)
    built_artifact = BuiltAIQArtifact(whl=wheel_data.whl_base64, metadata=metadata)

    return AIQArtifact(artifact=built_artifact, whl_path=wheel_data.whl_path)
