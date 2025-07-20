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

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from enum import Enum
from typing import Any

from aiq.data_models.component import ComponentGroup

logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """Component build state"""
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


@dataclasses.dataclass
class ComponentInfo:
    """Information about a component during building"""
    name: str
    config: Any
    state: ComponentState
    instance: Any | None = None
    error: Exception | None = None
    waiters: set[str] = dataclasses.field(default_factory=set)


class DependencyManager:
    """
    Manages component dependencies and build coordination for the WorkflowBuilder.

    This class handles:
    - Component state tracking and lifecycle management
    - Async dependency resolution and waiting
    - Build task coordination and cancellation
    - Thread-safe access to shared state
    """

    def __init__(self):
        # Component state tracking
        self._component_info: dict[str, ComponentInfo] = {}
        self._ready_events: dict[str, asyncio.Event] = {}
        self._build_tasks: dict[str, asyncio.Task] = {}

        # Configuration tracking for missing component detection
        self._configured_components: dict[str, ComponentGroup] = {}

        # Locks for thread-safe access to shared data structures
        self._component_info_lock = asyncio.Lock()
        self._ready_events_lock = asyncio.Lock()

    def register_component(self, component_name: str, component_group: ComponentGroup):
        """Register a component as configured in the system.

        Args:
            component_name (str): The name of the component to register.
            component_group (ComponentGroup): The group/type of the component.
        """
        self._configured_components[component_name] = component_group

    def is_component_defined(self, component_name: str) -> bool:
        """
        Check if a component is defined in the original configuration.
        This eliminates race conditions by checking against the source of truth.

        Args:
            component_name (str): The name of the component to check if it's defined

        Returns:
            bool: True if the component is defined, False otherwise
        """
        # Check if it's a special component
        if component_name == "<workflow>":
            return True

        # Check if it's in our configured components
        return component_name in self._configured_components

    async def wait_for_component(self, component_name: str, requester: str = "unknown") -> Any:
        """
        Wait for a component to be ready. If it's not ready, create an event and wait.

        Args:
            component_name (str): The name of the component to wait for
            requester (str, optional): The name of the component that's requesting this dependency.
                Defaults to "unknown".

        Returns:
            Any: The ready component instance (typically a configured wrapper containing both
                config and the actual component instance)

        Raises:
            RuntimeError: If component failed to build
            ValueError: If component is not defined in configuration
        """
        # Check if component is already ready (with lock protection)
        async with self._component_info_lock:
            if component_name in self._component_info:
                info = self._component_info[component_name]
                if info.state == ComponentState.READY:
                    return info.instance
                elif info.state == ComponentState.FAILED:
                    # Re-raise the original exception for backward compatibility
                    raise info.error or RuntimeError(f"Component '{component_name}' failed to build")

        # Check if component is defined in configuration (no race condition)
        if not self.is_component_defined(component_name):
            raise ValueError(f"Component `{component_name}` is not defined in your configuration. "
                             f"Make sure it's included in the appropriate section (functions, llms, etc.).")

        # Create event if it doesn't exist (with lock protection)
        async with self._ready_events_lock:
            if component_name not in self._ready_events:
                self._ready_events[component_name] = asyncio.Event()
            event = self._ready_events[component_name]

            # Track who's waiting for this component (with lock protection)
        async with self._component_info_lock:
            if component_name in self._component_info:
                # Add waiter FIRST, then check for cycles
                # This ensures we detect cycles even with concurrent waits
                self._component_info[component_name].waiters.add(requester)

                # Check for circular dependency after adding to waiters
                cycle = self._detect_dependency_cycle(component_name, requester)
                if cycle:
                    cycle_path = " → ".join(cycle)
                    raise ValueError(f"Circular dependency detected: {cycle_path}")

        logger.debug("Component %s waiting for %s", requester, component_name)

        # Wait for the component to be ready
        await event.wait()

        # Component should be ready now (with lock protection)
        async with self._component_info_lock:
            if component_name in self._component_info:
                info = self._component_info[component_name]
                if info.state == ComponentState.READY:
                    return info.instance
                elif info.state == ComponentState.FAILED:
                    # Re-raise the original exception for backward compatibility
                    raise info.error or RuntimeError(f"Component '{component_name}' failed to build")

        raise RuntimeError(f"Component {component_name} is not ready after waiting")

    async def mark_component_building(self, component_name: str, config: Any):
        """Mark a component as currently building.

        Args:
            component_name (str): The name of the component to mark as building
            config (Any): The configuration of the component
        """
        async with self._component_info_lock:
            self._component_info[component_name] = ComponentInfo(name=component_name,
                                                                 config=config,
                                                                 state=ComponentState.BUILDING)

    async def mark_component_ready(self, component_name: str, instance: Any):
        """
        Mark a component as ready and wake up any waiters.

        Args:
            component_name (str): The name of the component that's ready
            instance (Any): The ready component instance (typically a configured wrapper
                containing both config and the actual component instance)
        """
        # Update component info (with lock protection)
        async with self._component_info_lock:
            if component_name in self._component_info:
                self._component_info[component_name].state = ComponentState.READY
                self._component_info[component_name].instance = instance
            else:
                self._component_info[component_name] = ComponentInfo(name=component_name,
                                                                     config=None,
                                                                     state=ComponentState.READY,
                                                                     instance=instance)

        # Wake up any waiters (with lock protection)
        async with self._ready_events_lock:
            if component_name in self._ready_events:
                logger.debug("Component %s ready, waking up waiters", component_name)
                self._ready_events[component_name].set()

    async def mark_component_failed(self, component_name: str, error: Exception):
        """
        Mark a component as failed and wake up any waiters.

        Args:
            component_name (str): The name of the component that failed
            error (Exception): The error that caused the failure
        """
        # Ensure we always have a valid error
        if error is None:
            error = RuntimeError(f"Component '{component_name}' failed with unknown error")

        # Update component info (with lock protection)
        async with self._component_info_lock:
            if component_name in self._component_info:
                self._component_info[component_name].state = ComponentState.FAILED
                self._component_info[component_name].error = error
            else:
                self._component_info[component_name] = ComponentInfo(name=component_name,
                                                                     config=None,
                                                                     state=ComponentState.FAILED,
                                                                     error=error)

        # Wake up any waiters (they will get the error) (with lock protection)
        async with self._ready_events_lock:
            if component_name in self._ready_events:
                logger.debug("Component %s failed, waking up waiters", component_name)
                self._ready_events[component_name].set()

    def get_component_state(self, component_name: str) -> ComponentState | None:
        """Get the current state of a component, if it exists.

        Args:
            component_name (str): The name of the component to check.

        Returns:
            ComponentState | None: The current state of the component, or None if not found.
        """
        if component_name in self._component_info:
            return self._component_info[component_name].state
        return None

    def get_component_error(self, component_name: str) -> Exception | None:
        """Get the error for a failed component, if it exists.

        Args:
            component_name (str): The name of the component to get the error for

        Returns:
            Exception | None: The error for the component, if it exists
        """
        if component_name in self._component_info:
            return self._component_info[component_name].error
        return None

    def _detect_dependency_cycle(self, component_name: str, requester: str) -> list[str] | None:
        """
        Detect if adding this dependency would create a cycle using simple DFS.

        Args:
            component_name (str): The component being waited for
            requester (str): The component that wants to wait

        Returns:
            list[str] | None: List representing the cycle path if found, None otherwise
        """

        # Simple question: If requester waits for component_name,
        # can we eventually get back to requester by following the waiting chain?

        def can_reach(start: str, target: str, visited: set[str]) -> list[str] | None:
            if start == target:
                return [start]  # Found the target!

            if start in visited:
                return None  # Already explored

            visited.add(start)

            # Who is 'start' waiting for? Look through all components
            # to see if 'start' is in their waiters
            for comp_name, comp_info in self._component_info.items():
                if start in comp_info.waiters:
                    # 'start' is waiting for 'comp_name'
                    path = can_reach(comp_name, target, visited.copy())
                    if path:
                        return [start] + path

            return None

        # Check: if requester waits for component_name, can component_name eventually reach requester?
        cycle_path = can_reach(component_name, requester, set())

        if cycle_path:
            # We found a cycle! Format it nicely
            return [requester] + cycle_path
        else:
            return None

    def get_component_instance(self, component_name: str) -> Any | None:
        """Get the instance of a ready component, if available.

        Args:
            component_name (str): The name of the component to get the instance for.

        Returns:
            Any | None: The component instance if ready (typically a configured wrapper
                containing both config and the actual component instance), None otherwise.
        """
        if component_name in self._component_info:
            info = self._component_info[component_name]
            if info.state == ComponentState.READY:
                return info.instance
        return None

    def register_build_task(self, component_name: str, task: asyncio.Task):
        """Register a build task for tracking and cancellation.

        Args:
            component_name (str): The name of the component being built.
            task (asyncio.Task): The asyncio task building the component.
        """
        self._build_tasks[component_name] = task

    def start_component_build(self, component_name: str, build_fn: Callable[[], Any]) -> asyncio.Task:
        """Start building a component asynchronously and register the task for coordination.

        Args:
            component_name (str): Name of the component to build
            build_fn (Callable[[], Any]): Async function that builds the component

        Returns:
            asyncio.Task: The created asyncio task
        """
        task = asyncio.create_task(build_fn())
        self.register_build_task(component_name, task)
        return task

    async def cancel_build_task(self, component_name: str):
        """Cancel a specific build task if it exists and is running.

        Args:
            component_name (str): The name of the component whose build task to cancel.
        """
        if component_name in self._build_tasks:
            task = self._build_tasks[component_name]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self._build_tasks[component_name]

    async def cancel_all_build_tasks(self):
        """Cancel all running build tasks."""
        # Cancel any running build tasks
        for task in self._build_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete or be cancelled
        if self._build_tasks:
            await asyncio.gather(*self._build_tasks.values(), return_exceptions=True)

        self._build_tasks.clear()

    def clear_component_state(self, component_name: str):
        """Clear all state for a specific component (for rebuild scenarios).

        Args:
            component_name (str): The name of the component to clear state for.
        """
        # Remove from component info
        if component_name in self._component_info:
            del self._component_info[component_name]

        # Remove ready event
        if component_name in self._ready_events:
            del self._ready_events[component_name]

    async def build_component_with_coordination(self,
                                                component_name: str,
                                                component_group: ComponentGroup,
                                                config: Any,
                                                build_fn: Callable[[str, Any], Awaitable[Any]]):
        """
        Build a component with full dependency coordination and error handling.

        Args:
            component_name (str): Name of the component to build
            component_group (ComponentGroup): Type/group of the component
            config (Any): Component configuration
            build_fn (Callable[[str, Any], Awaitable[Any]]): Async function that builds the
                component and returns a configured wrapper containing both config and instance
        """
        from aiq.builder.exceptions import DependencyNotReadyError

        try:
            # Mark as building
            await self.mark_component_building(component_name, config)

            logger.debug("Starting to build %s component: %s", component_group.value, component_name)

            # Build the component
            instance = await build_fn(component_name, config)

            # Mark as ready
            await self.mark_component_ready(component_name, instance)
            logger.debug("Successfully built %s component: %s", component_group.value, component_name)

        except DependencyNotReadyError as e:
            # Wait for the dependency and retry once (no loop needed)
            logger.debug("Component %s waiting for dependency %s", component_name, e.dependency_name)
            await self.wait_for_component(e.dependency_name, component_name)

            # Retry the build once after dependency is ready
            await self.build_component_with_coordination(component_name, component_group, config, build_fn)

        except Exception as e:
            # Handle all other errors
            logger.error("Failed to build %s component %s: %s", component_group.value, component_name, e, exc_info=True)
            await self.mark_component_failed(component_name, e)
