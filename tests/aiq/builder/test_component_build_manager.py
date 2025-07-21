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

# type: ignore[name-defined]

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from aiq.builder.component_build_manager import ComponentBuildManager
from aiq.builder.component_build_manager import ComponentInfo
from aiq.builder.component_build_manager import ComponentState
from aiq.builder.exceptions import DependencyNotReadyError
from aiq.data_models.component import ComponentGroup


class TestComponentBuildManager:
    """Tests for ComponentBuildManager class"""

    @pytest.fixture
    def manager(self):
        """Create a fresh ComponentBuildManager instance for each test"""
        return ComponentBuildManager()

    @pytest.fixture
    def mock_config(self):
        """Mock configuration object"""
        return Mock()

    @pytest.fixture
    def mock_instance(self):
        """Mock component instance"""
        return Mock()

    def test_init(self, manager):
        """Test ComponentBuildManager initialization"""
        assert len(manager._component_info) == 0
        assert len(manager._ready_events) == 0
        assert len(manager._build_tasks) == 0
        assert len(manager._configured_components) == 0
        assert manager._component_info_lock is not None
        assert manager._ready_events_lock is not None

    def test_register_component(self, manager):
        """Test registering components"""
        manager.register_component("test_component", ComponentGroup.FUNCTIONS)
        assert "test_component" in manager._configured_components
        assert manager._configured_components["test_component"] == ComponentGroup.FUNCTIONS

        # Test registering different component groups
        manager.register_component("test_llm", ComponentGroup.LLMS)
        manager.register_component("test_embedder", ComponentGroup.EMBEDDERS)
        manager.register_component("test_memory", ComponentGroup.MEMORY)
        manager.register_component("test_retriever", ComponentGroup.RETRIEVERS)
        manager.register_component("test_tracing", ComponentGroup.TRACING)

        assert len(manager._configured_components) == 6
        assert manager._configured_components["test_llm"] == ComponentGroup.LLMS
        assert manager._configured_components["test_embedder"] == ComponentGroup.EMBEDDERS
        assert manager._configured_components["test_memory"] == ComponentGroup.MEMORY
        assert manager._configured_components["test_retriever"] == ComponentGroup.RETRIEVERS
        assert manager._configured_components["test_tracing"] == ComponentGroup.TRACING

    def test_is_component_defined(self, manager):
        """Test checking if a component is defined"""
        # Special workflow component should always be defined
        assert manager.is_component_defined("<workflow>") is True

        # Non-registered component should not be defined
        assert manager.is_component_defined("non_existent") is False

        # Registered component should be defined
        manager.register_component("test_component", ComponentGroup.FUNCTIONS)
        assert manager.is_component_defined("test_component") is True

    async def test_mark_component_building(self, manager, mock_config):
        """Test marking a component as building"""
        await manager.mark_component_building("test_component", mock_config)

        assert "test_component" in manager._component_info
        info = manager._component_info["test_component"]
        assert info.name == "test_component"
        assert info.config == mock_config
        assert info.state == ComponentState.BUILDING
        assert info.instance is None
        assert info.error is None

    async def test_mark_component_building_preserves_waiters(self, manager, mock_config):
        """Test that marking component as building preserves existing waiters"""
        # Create a component info with waiters first
        manager._component_info["test_component"] = ComponentInfo(name="test_component",
                                                                  config=None,
                                                                  state=ComponentState.PENDING,
                                                                  waiters={"waiter1", "waiter2"})

        await manager.mark_component_building("test_component", mock_config)

        info = manager._component_info["test_component"]
        assert info.state == ComponentState.BUILDING
        assert info.config == mock_config
        assert info.waiters == {"waiter1", "waiter2"}  # Waiters should be preserved

    async def test_mark_component_ready(self, manager, mock_instance):
        """Test marking a component as ready"""
        await manager.mark_component_ready("test_component", mock_instance)

        assert "test_component" in manager._component_info
        info = manager._component_info["test_component"]
        assert info.name == "test_component"
        assert info.state == ComponentState.READY
        assert info.instance == mock_instance

    async def test_mark_component_ready_with_existing_info(self, manager, mock_config, mock_instance):
        """Test marking a component as ready when it already has info"""
        # First mark as building
        await manager.mark_component_building("test_component", mock_config)

        # Then mark as ready
        await manager.mark_component_ready("test_component", mock_instance)

        info = manager._component_info["test_component"]
        assert info.name == "test_component"
        assert info.config == mock_config  # Config should be preserved
        assert info.state == ComponentState.READY
        assert info.instance == mock_instance

    async def test_mark_component_failed(self, manager):
        """Test marking a component as failed"""
        test_error = ValueError("Test error")
        await manager.mark_component_failed("test_component", test_error)

        assert "test_component" in manager._component_info
        info = manager._component_info["test_component"]
        assert info.name == "test_component"
        assert info.state == ComponentState.FAILED
        assert info.error == test_error
        assert info.instance is None

    async def test_mark_component_failed_with_none_error(self, manager):
        """Test marking a component as failed with None error"""
        await manager.mark_component_failed("test_component", None)

        info = manager._component_info["test_component"]
        assert info.state == ComponentState.FAILED
        assert isinstance(info.error, RuntimeError)
        assert "test_component" in str(info.error)

    def test_get_component_state(self, manager, mock_config):
        """Test getting component state"""
        # Non-existent component should return None
        assert manager.get_component_state("non_existent") is None

        # Existing component should return correct state
        manager._component_info["test_component"] = ComponentInfo(name="test_component",
                                                                  config=mock_config,
                                                                  state=ComponentState.BUILDING)
        assert manager.get_component_state("test_component") == ComponentState.BUILDING

    def test_get_component_error(self, manager):
        """Test getting component error"""
        # Non-existent component should return None
        assert manager.get_component_error("non_existent") is None

        # Component with error should return the error
        test_error = ValueError("Test error")
        manager._component_info["test_component"] = ComponentInfo(name="test_component",
                                                                  config=None,
                                                                  state=ComponentState.FAILED,
                                                                  error=test_error)
        assert manager.get_component_error("test_component") == test_error

    def test_get_component_instance(self, manager, mock_instance):
        """Test getting component instance"""
        # Non-existent component should return None
        assert manager.get_component_instance("non_existent") is None

        # Component not ready should return None
        manager._component_info["building_component"] = ComponentInfo(name="building_component",
                                                                      config=None,
                                                                      state=ComponentState.BUILDING)
        assert manager.get_component_instance("building_component") is None

        # Ready component should return instance
        manager._component_info["ready_component"] = ComponentInfo(name="ready_component",
                                                                   config=None,
                                                                   state=ComponentState.READY,
                                                                   instance=mock_instance)
        assert manager.get_component_instance("ready_component") == mock_instance

    def test_register_build_task(self, manager):
        """Test registering build tasks"""
        mock_task = AsyncMock()
        manager.register_build_task("test_component", mock_task)
        assert manager._build_tasks["test_component"] == mock_task

    async def test_start_component_build(self, manager):
        """Test starting component build"""
        mock_build_fn = AsyncMock(return_value="result")

        task = manager.start_component_build("test_component", mock_build_fn)

        assert isinstance(task, asyncio.Task)
        assert "test_component" in manager._build_tasks
        assert manager._build_tasks["test_component"] == task

        # Clean up the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_cancel_build_task(self, manager):
        """Test canceling a build task"""

        # Create a real task instead of a mock
        async def dummy_task():
            await asyncio.sleep(10)  # Long sleep to ensure it doesn't complete

        task = asyncio.create_task(dummy_task())
        manager._build_tasks["test_component"] = task

        await manager.cancel_build_task("test_component")

        # Task should be cancelled
        assert task.cancelled()
        assert "test_component" not in manager._build_tasks

    async def test_cancel_build_task_already_done(self, manager):
        """Test canceling a build task that's already done"""

        # Create a task that completes immediately
        async def quick_task():
            return "done"

        task = asyncio.create_task(quick_task())
        await task  # Wait for it to complete
        manager._build_tasks["test_component"] = task

        await manager.cancel_build_task("test_component")

        # Should not try to cancel a done task, just remove it
        assert not task.cancelled()  # It completed, not cancelled
        assert "test_component" not in manager._build_tasks

    async def test_cancel_all_build_tasks(self, manager):
        """Test canceling all build tasks"""

        # Create real tasks instead of mocks
        async def dummy_task1():
            await asyncio.sleep(10)

        async def dummy_task2():
            await asyncio.sleep(10)

        task1 = asyncio.create_task(dummy_task1())
        task2 = asyncio.create_task(dummy_task2())

        manager._build_tasks["component1"] = task1
        manager._build_tasks["component2"] = task2

        await manager.cancel_all_build_tasks()

        # Both tasks should be cancelled
        assert task1.cancelled()
        assert task2.cancelled()
        assert len(manager._build_tasks) == 0

    def test_clear_component_state(self, manager, mock_config):
        """Test clearing component state"""
        # Set up some state
        manager._component_info["test_component"] = ComponentInfo(name="test_component",
                                                                  config=mock_config,
                                                                  state=ComponentState.READY)
        manager._ready_events["test_component"] = asyncio.Event()

        # Clear the state
        manager.clear_component_state("test_component")

        assert "test_component" not in manager._component_info
        assert "test_component" not in manager._ready_events

    def test_detect_dependency_cycle_no_cycle(self, manager):
        """Test cycle detection when there's no cycle"""
        # Set up a simple dependency chain: A -> B -> C
        manager._component_info = {
            "A": ComponentInfo(name="A", config=None, state=ComponentState.PENDING, waiters=set()),
            "B": ComponentInfo(name="B", config=None, state=ComponentState.PENDING, waiters={"A"}),
            "C": ComponentInfo(name="C", config=None, state=ComponentState.PENDING, waiters={"B"})
        }

        # Adding D -> A should not create a cycle
        cycle = manager._detect_dependency_cycle("A", "D")
        assert cycle is None

    def test_detect_dependency_cycle_with_cycle(self, manager):
        """Test cycle detection when there's a cycle"""
        # Set up a cycle: A waits for B, B waits for C
        manager._component_info = {
            "A": ComponentInfo(name="A", config=None, state=ComponentState.PENDING, waiters=set()),
            "B": ComponentInfo(name="B", config=None, state=ComponentState.PENDING, waiters={"A"}),
            "C": ComponentInfo(name="C", config=None, state=ComponentState.PENDING, waiters={"B"})
        }

        # Now if C tries to wait for A, it creates a cycle: A -> B -> C -> A
        cycle = manager._detect_dependency_cycle("A", "C")
        assert cycle is not None
        assert "A" in cycle
        assert "B" in cycle
        assert "C" in cycle
        assert cycle[0] == "C"  # Requester should be first
        # The cycle detection algorithm returns the path, last element depends on implementation

    def test_detect_dependency_cycle_self_dependency(self, manager):
        """Test cycle detection for self-dependency"""
        manager._component_info = {
            "A": ComponentInfo(name="A", config=None, state=ComponentState.PENDING, waiters=set())
        }

        # A trying to wait for itself
        cycle = manager._detect_dependency_cycle("A", "A")
        assert cycle is not None
        assert cycle == ["A", "A"]

    async def test_wait_for_component_already_ready(self, manager, mock_instance):
        """Test waiting for a component that's already ready"""
        # Set up a ready component
        manager._component_info["test_component"] = ComponentInfo(name="test_component",
                                                                  config=None,
                                                                  state=ComponentState.READY,
                                                                  instance=mock_instance)

        result = await manager.wait_for_component("test_component", "requester")
        assert result == mock_instance

    async def test_wait_for_component_already_failed(self, manager):
        """Test waiting for a component that's already failed"""
        test_error = ValueError("Test error")
        manager._component_info["test_component"] = ComponentInfo(name="test_component",
                                                                  config=None,
                                                                  state=ComponentState.FAILED,
                                                                  error=test_error)

        with pytest.raises(ValueError, match="Test error"):
            await manager.wait_for_component("test_component", "requester")

    async def test_wait_for_component_not_defined(self, manager):
        """Test waiting for a component that's not defined"""
        with pytest.raises(ValueError, match="is not defined in your configuration"):
            await manager.wait_for_component("undefined_component", "requester")

    async def test_wait_for_component_circular_dependency(self, manager):
        """Test waiting for a component that would create a circular dependency"""
        # Set up the manager to think components are defined
        manager.register_component("A", ComponentGroup.FUNCTIONS)
        manager.register_component("B", ComponentGroup.FUNCTIONS)

        # Set up A waiting for B
        manager._component_info = {
            "A": ComponentInfo(name="A", config=None, state=ComponentState.PENDING, waiters=set()),
            "B": ComponentInfo(name="B", config=None, state=ComponentState.PENDING, waiters={"A"})
        }

        # B trying to wait for A should detect the cycle
        with pytest.raises(ValueError, match="Circular dependency detected"):
            await manager.wait_for_component("A", "B")

    async def test_wait_for_component_becomes_ready(self, manager, mock_instance):
        """Test waiting for a component that becomes ready during the wait"""
        manager.register_component("test_component", ComponentGroup.FUNCTIONS)

        async def mark_ready_after_delay():
            await asyncio.sleep(0.1)  # Small delay
            await manager.mark_component_ready("test_component", mock_instance)

        # Start the task that will mark the component ready
        ready_task = asyncio.create_task(mark_ready_after_delay())

        # Wait for the component
        result = await manager.wait_for_component("test_component", "requester")

        await ready_task  # Clean up the task
        assert result == mock_instance

    async def test_wait_for_component_becomes_failed(self, manager):
        """Test waiting for a component that becomes failed during the wait"""
        manager.register_component("test_component", ComponentGroup.FUNCTIONS)
        test_error = RuntimeError("Build failed")

        async def mark_failed_after_delay():
            await asyncio.sleep(0.1)  # Small delay
            await manager.mark_component_failed("test_component", test_error)

        # Start the task that will mark the component failed
        fail_task = asyncio.create_task(mark_failed_after_delay())

        # Wait for the component - should raise the error
        with pytest.raises(RuntimeError, match="Build failed"):
            await manager.wait_for_component("test_component", "requester")

        await fail_task  # Clean up the task

    async def test_build_component_with_coordination_success(self, manager, mock_config, mock_instance):
        """Test successful component building with coordination"""

        async def mock_build_fn(name, config):
            assert name == "test_component"
            assert config == mock_config
            return mock_instance

        await manager.build_component_with_coordination("test_component",
                                                        ComponentGroup.FUNCTIONS,
                                                        mock_config,
                                                        mock_build_fn)

        # Check that component is marked as ready
        info = manager._component_info["test_component"]
        assert info.state == ComponentState.READY
        assert info.instance == mock_instance
        assert info.config == mock_config

    async def test_build_component_with_coordination_dependency_not_ready(self, manager, mock_config):
        """Test component building with dependency not ready error"""
        dependency_error = DependencyNotReadyError("Dependency not ready", "dependency_component")
        call_count = 0

        async def mock_build_fn(name, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise dependency_error
            else:
                return "success_instance"

        # Mock the wait_for_component method to simulate dependency becoming ready
        manager.wait_for_component = AsyncMock(return_value="dependency_instance")

        await manager.build_component_with_coordination("test_component",
                                                        ComponentGroup.FUNCTIONS,
                                                        mock_config,
                                                        mock_build_fn)

        # Should have called wait_for_component
        manager.wait_for_component.assert_called_once_with("dependency_component", "test_component")

        # Should have retried and succeeded
        assert call_count == 2
        info = manager._component_info["test_component"]
        assert info.state == ComponentState.READY
        assert info.instance == "success_instance"

    async def test_build_component_with_coordination_other_error(self, manager, mock_config):
        """Test component building with other type of error"""
        build_error = RuntimeError("Build failed")

        async def mock_build_fn(name, config):
            raise build_error

        await manager.build_component_with_coordination("test_component",
                                                        ComponentGroup.FUNCTIONS,
                                                        mock_config,
                                                        mock_build_fn)

        # Should mark component as failed
        info = manager._component_info["test_component"]
        assert info.state == ComponentState.FAILED
        assert info.error == build_error

    async def test_concurrent_wait_for_same_component(self, manager, mock_instance):
        """Test multiple tasks waiting for the same component concurrently"""
        manager.register_component("test_component", ComponentGroup.FUNCTIONS)

        # Start multiple wait tasks
        wait_tasks = [
            asyncio.create_task(manager.wait_for_component("test_component", f"requester_{i}")) for i in range(5)
        ]

        # Let them start waiting
        await asyncio.sleep(0.1)

        # Mark component as ready
        await manager.mark_component_ready("test_component", mock_instance)

        # All wait tasks should complete with the same instance
        results = await asyncio.gather(*wait_tasks)
        assert all(result == mock_instance for result in results)

    async def test_concurrent_component_operations(self, manager, mock_config, mock_instance):
        """Test concurrent operations on different components"""
        manager.register_component("component1", ComponentGroup.FUNCTIONS)
        manager.register_component("component2", ComponentGroup.LLMS)
        manager.register_component("component3", ComponentGroup.EMBEDDERS)

        # Simulate concurrent operations
        tasks = [
            asyncio.create_task(manager.mark_component_building("component1", mock_config)),
            asyncio.create_task(manager.mark_component_ready("component2", mock_instance)),
            asyncio.create_task(manager.mark_component_failed("component3", ValueError("Error"))),
        ]

        await asyncio.gather(*tasks)

        # Check all operations completed correctly
        assert manager.get_component_state("component1") == ComponentState.BUILDING
        assert manager.get_component_state("component2") == ComponentState.READY
        assert manager.get_component_state("component3") == ComponentState.FAILED
        assert manager.get_component_instance("component2") == mock_instance
        assert isinstance(manager.get_component_error("component3"), ValueError)

    async def test_component_waiter_tracking(self, manager):
        """Test that component waiters are tracked correctly"""
        manager.register_component("test_component", ComponentGroup.FUNCTIONS)

        # Create a wait task (this should add to waiters)
        wait_task = asyncio.create_task(manager.wait_for_component("test_component", "requester1"))

        # Give it a moment to set up
        await asyncio.sleep(0.1)

        # Check that the waiter was added
        info = manager._component_info["test_component"]
        assert "requester1" in info.waiters

        # Mark component ready to complete the wait
        await manager.mark_component_ready("test_component", "instance")

        # Clean up
        await wait_task

    def test_component_state_enum_values(self):
        """Test that ComponentState enum has expected values"""
        assert ComponentState.PENDING.value == "pending"
        assert ComponentState.BUILDING.value == "building"
        assert ComponentState.READY.value == "ready"
        assert ComponentState.FAILED.value == "failed"

    def test_component_info_dataclass(self, mock_config, mock_instance):
        """Test ComponentInfo dataclass functionality"""
        # Test with minimal fields
        info = ComponentInfo(name="test", config=mock_config, state=ComponentState.PENDING)
        assert info.name == "test"
        assert info.config == mock_config
        assert info.state == ComponentState.PENDING
        assert info.instance is None
        assert info.error is None
        assert len(info.waiters) == 0

        # Test with all fields
        test_error = ValueError("Test error")
        full_info = ComponentInfo(name="full_test",
                                  config=mock_config,
                                  state=ComponentState.FAILED,
                                  instance=mock_instance,
                                  error=test_error,
                                  waiters={"waiter1", "waiter2"})
        assert full_info.name == "full_test"
        assert full_info.config == mock_config
        assert full_info.state == ComponentState.FAILED
        assert full_info.instance == mock_instance
        assert full_info.error == test_error
        assert full_info.waiters == {"waiter1", "waiter2"}

    async def test_mark_component_ready_wakes_waiters(self, manager, mock_instance):
        """Test that marking a component ready wakes up waiters"""
        manager.register_component("test_component", ComponentGroup.FUNCTIONS)

        # Create multiple wait tasks
        wait_tasks = [
            asyncio.create_task(manager.wait_for_component("test_component", f"requester_{i}")) for i in range(3)
        ]

        # Let them start waiting
        await asyncio.sleep(0.1)

        # Verify that there are waiters
        assert len(manager._component_info["test_component"].waiters) == 3

        # Mark component as ready - this should wake all waiters
        await manager.mark_component_ready("test_component", mock_instance)

        # All tasks should complete quickly now
        results = await asyncio.wait_for(asyncio.gather(*wait_tasks), timeout=1.0)
        assert all(result == mock_instance for result in results)

    async def test_mark_component_failed_wakes_waiters(self, manager):
        """Test that marking a component failed wakes up waiters with the error"""
        manager.register_component("test_component", ComponentGroup.FUNCTIONS)
        test_error = RuntimeError("Build failed")

        # Create multiple wait tasks
        wait_tasks = [
            asyncio.create_task(manager.wait_for_component("test_component", f"requester_{i}")) for i in range(3)
        ]

        # Let them start waiting
        await asyncio.sleep(0.1)

        # Mark component as failed - this should wake all waiters with the error
        await manager.mark_component_failed("test_component", test_error)

        # All tasks should fail with the same error
        for task in wait_tasks:
            with pytest.raises(RuntimeError, match="Build failed"):
                await task

    async def test_complex_dependency_chain(self, manager):
        """Test a complex but valid dependency chain"""
        # Register components
        for comp in ["A", "B", "C", "D"]:
            manager.register_component(comp, ComponentGroup.FUNCTIONS)

        # Set up dependency chain: A -> B -> C -> D
        manager._component_info = {
            "A": ComponentInfo(name="A", config=None, state=ComponentState.PENDING, waiters=set()),
            "B": ComponentInfo(name="B", config=None, state=ComponentState.PENDING, waiters={"A"}),
            "C": ComponentInfo(name="C", config=None, state=ComponentState.PENDING, waiters={"B"}),
            "D": ComponentInfo(name="D", config=None, state=ComponentState.PENDING, waiters={"C"}),
        }

        # This should not detect a cycle
        cycle = manager._detect_dependency_cycle("B", "A")
        assert cycle is None

        # But creating a cycle should be detected
        cycle = manager._detect_dependency_cycle("A", "D")
        assert cycle is not None
        # The cycle includes D -> C -> B -> A
        assert len(cycle) >= 4  # Should have at least 4 elements in the cycle
