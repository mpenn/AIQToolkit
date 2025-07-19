# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class DependencyNotReadyError(Exception):
    """
    Exception raised when a dependency is not ready yet during component building.

    This is not an actual error, but a signal that the component should wait
    for the dependency to be ready. This helps distinguish between actual errors
    and normal building flow.
    """

    def __init__(self, message: str, dependency_name: str):
        super().__init__(message)
        self.dependency_name = dependency_name
