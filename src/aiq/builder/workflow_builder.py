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
import inspect
import logging
import warnings
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from typing import Any

from aiq.builder.builder import Builder
from aiq.builder.builder import UserManagerHolder
from aiq.builder.context import AIQContext
from aiq.builder.context import AIQContextState
from aiq.builder.dependency_manager import ComponentState
from aiq.builder.dependency_manager import DependencyManager
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function import Function
from aiq.builder.function import LambdaFunction
from aiq.builder.function_info import FunctionInfo
from aiq.builder.llm import LLMProviderInfo
from aiq.builder.retriever import RetrieverProviderInfo
from aiq.builder.workflow import Workflow
from aiq.cli.type_registry import GlobalTypeRegistry
from aiq.cli.type_registry import TypeRegistry
from aiq.data_models.component import ComponentGroup
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.component_ref import MemoryRef
from aiq.data_models.component_ref import RetrieverRef
from aiq.data_models.config import AIQConfig
from aiq.data_models.config import GeneralConfig
from aiq.data_models.embedder import EmbedderBaseConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.function_dependencies import FunctionDependencies
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.logging import LoggingBaseConfig
from aiq.data_models.memory import MemoryBaseConfig
from aiq.data_models.retriever import RetrieverBaseConfig
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from aiq.memory.interfaces import MemoryEditor
from aiq.observability.exporter.base_exporter import BaseExporter
from aiq.profiler.decorators.framework_wrapper import chain_wrapped_build_fn
from aiq.profiler.utils import detect_llm_frameworks_in_build_fn
from aiq.utils.type_utils import override

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConfiguredTelemetryExporter:
    config: TelemetryExporterBaseConfig
    instance: BaseExporter


@dataclasses.dataclass
class ConfiguredFunction:
    config: FunctionBaseConfig
    instance: Function


@dataclasses.dataclass
class ConfiguredLLM:
    config: LLMBaseConfig
    instance: LLMProviderInfo


@dataclasses.dataclass
class ConfiguredEmbedder:
    config: EmbedderBaseConfig
    instance: EmbedderProviderInfo


@dataclasses.dataclass
class ConfiguredMemory:
    config: MemoryBaseConfig
    instance: MemoryEditor


@dataclasses.dataclass
class ConfiguredRetriever:
    config: RetrieverBaseConfig
    instance: RetrieverProviderInfo


class WorkflowBuilder(Builder, AbstractAsyncContextManager):
    """
    A workflow builder that provides both synchronous and asynchronous interfaces
    for backwards compatibility while using dynamic dependency resolution internally.

    All components are built in parallel and pause when they need dependencies
    that aren't ready yet, using asyncio events for coordination managed by DependencyManager.
    """

    def __init__(self, *, general_config: GeneralConfig | None = None, registry: TypeRegistry | None = None):
        if general_config is None:
            general_config = GeneralConfig()

        if registry is None:
            registry = GlobalTypeRegistry.get()

        self.general_config = general_config
        self._registry = registry

        # Dependency management - the core of this refactor
        self._dependency_manager = DependencyManager()

        # Storage for completed components
        self._functions: dict[str, ConfiguredFunction] = {}
        self._workflow: ConfiguredFunction | None = None
        self._llms: dict[str, ConfiguredLLM] = {}
        self._embedders: dict[str, ConfiguredEmbedder] = {}
        self._memory_clients: dict[str, ConfiguredMemory] = {}
        self._retrievers: dict[str, ConfiguredRetriever] = {}
        self._telemetry_exporters: dict[str, ConfiguredTelemetryExporter] = {}
        self._logging_handlers: dict[str, logging.Handler] = {}

        # Locks for thread-safe access to shared data structures
        self._logging_handlers_lock = asyncio.Lock()
        self._telemetry_exporters_lock = asyncio.Lock()

        # Context and dependencies
        self._context_state = AIQContextState.get()
        self._exit_stack: AsyncExitStack | None = None
        self.function_dependencies: dict[str, FunctionDependencies] = {}
        self.current_function_building: str | None = None

    async def __aenter__(self):
        self._exit_stack = AsyncExitStack()

        # Set up logging handlers (needed early for build process logging)
        telemetry_config = self.general_config.telemetry
        logger.info("Building %s telemetry logging handlers", len(telemetry_config.logging))
        await asyncio.gather(
            *
            [self.add_logging_handler(key, logging_config) for key, logging_config in telemetry_config.logging.items()])

        # Note: Telemetry exporters are now built in parallel with other components
        # in populate_builder for better performance

        return self

    async def __aexit__(self, *exc_details):
        assert self._exit_stack is not None, "Exit stack not initialized"

        # Use dependency manager to cancel all build tasks
        await self._dependency_manager.cancel_all_build_tasks()

        # Clean up logging handlers
        async with self._logging_handlers_lock:
            for _, handler in self._logging_handlers.items():
                logging.getLogger().removeHandler(handler)

        await self._exit_stack.__aexit__(*exc_details)

    def _get_exit_stack(self) -> AsyncExitStack:
        if self._exit_stack is None:
            raise ValueError(
                "Exit stack not initialized. Did you forget to call `async with WorkflowBuilder() as builder`?")
        return self._exit_stack

    # Internal build methods - these are the actual component builders
    async def _build_llm_internal(self, name: str, config: LLMBaseConfig) -> ConfiguredLLM:
        """Internal LLM building method"""
        llm_info = self._registry.get_llm_provider(type(config))
        info_obj = await self._get_exit_stack().enter_async_context(llm_info.build_fn(config, self))
        configured_llm = ConfiguredLLM(config=config, instance=info_obj)
        self._llms[name] = configured_llm
        return configured_llm

    async def _build_embedder_internal(self, name: str, config: EmbedderBaseConfig) -> ConfiguredEmbedder:
        """Internal embedder building method"""
        embedder_info = self._registry.get_embedder_provider(type(config))
        info_obj = await self._get_exit_stack().enter_async_context(embedder_info.build_fn(config, self))
        configured_embedder = ConfiguredEmbedder(config=config, instance=info_obj)
        self._embedders[name] = configured_embedder
        return configured_embedder

    async def _build_memory_internal(self, name: str, config: MemoryBaseConfig) -> ConfiguredMemory:
        """Internal memory building method"""
        memory_info = self._registry.get_memory(type(config))
        info_obj = await self._get_exit_stack().enter_async_context(memory_info.build_fn(config, self))
        configured_memory = ConfiguredMemory(config=config, instance=info_obj)
        self._memory_clients[name] = configured_memory
        return configured_memory

    async def _build_retriever_internal(self, name: str, config: RetrieverBaseConfig) -> ConfiguredRetriever:
        """Internal retriever building method"""
        retriever_info = self._registry.get_retriever_provider(type(config))
        info_obj = await self._get_exit_stack().enter_async_context(retriever_info.build_fn(config, self))
        configured_retriever = ConfiguredRetriever(config=config, instance=info_obj)
        self._retrievers[name] = configured_retriever
        return configured_retriever

    async def _build_function_internal(self, name: str, config: FunctionBaseConfig) -> ConfiguredFunction:
        """Internal function building method"""
        registration = self._registry.get_function(type(config))
        inner_builder = AsyncChildBuilder(self, requester_name=name)

        llms = {k: v.instance for k, v in self._llms.items()}
        function_frameworks = detect_llm_frameworks_in_build_fn(registration)
        build_fn = chain_wrapped_build_fn(registration.build_fn, llms, function_frameworks)

        # Track function dependencies
        self.current_function_building = config.type
        self.function_dependencies[config.type] = FunctionDependencies()

        build_result = await self._get_exit_stack().enter_async_context(build_fn(config, inner_builder))
        self.function_dependencies[name] = inner_builder.dependencies

        # Process build result
        if inspect.isfunction(build_result):
            build_result = FunctionInfo.from_fn(build_result)

        if isinstance(build_result, FunctionInfo):
            build_result = LambdaFunction.from_info(config=config, info=build_result, instance_name=name)

        if not isinstance(build_result, Function):
            raise ValueError("Expected a function, FunctionInfo object, or FunctionBase object to be "
                             f"returned from the function builder. Got {type(build_result)}")

        configured_function = ConfiguredFunction(config=config, instance=build_result)

        # Handle the special case of the workflow
        if name == "<workflow>":
            self._workflow = configured_function
        else:
            self._functions[name] = configured_function

        return configured_function

    async def _build_telemetry_exporter_internal(self, name: str,
                                                 config: TelemetryExporterBaseConfig) -> ConfiguredTelemetryExporter:
        """Internal telemetry exporter building method"""
        exporter_info = self._registry.get_telemetry_exporter(type(config))
        exporter = await self._get_exit_stack().enter_async_context(exporter_info.build_fn(config, self))
        configured_exporter = ConfiguredTelemetryExporter(config=config, instance=exporter)
        self._telemetry_exporters[name] = configured_exporter
        return configured_exporter

    # Component building wrapper that uses DependencyManager
    async def _build_component_coordinated(self, component_name: str, component_group: ComponentGroup, config: Any):
        """Build a component using dependency manager coordination."""

        async def build_fn(name: str, cfg: Any) -> Any:
            """Inner build function that calls the appropriate internal builder."""
            match component_group:
                case ComponentGroup.LLMS:
                    return await self._build_llm_internal(name, cfg)
                case ComponentGroup.EMBEDDERS:
                    return await self._build_embedder_internal(name, cfg)
                case ComponentGroup.MEMORY:
                    return await self._build_memory_internal(name, cfg)
                case ComponentGroup.RETRIEVERS:
                    return await self._build_retriever_internal(name, cfg)
                case ComponentGroup.FUNCTIONS:
                    return await self._build_function_internal(name, cfg)
                case ComponentGroup.TRACING:
                    return await self._build_telemetry_exporter_internal(name, cfg)
                case _:
                    raise ValueError(f"Unknown component group: {component_group}")

        await self._dependency_manager.build_component_with_coordination(component_name,
                                                                         component_group,
                                                                         config,
                                                                         build_fn)

    # Backwards compatible synchronous interface using DependencyManager
    @override
    async def add_function(self, name: str | FunctionRef, config: FunctionBaseConfig) -> Function:
        """Add a function and start building it immediately"""
        if name in self._functions:
            raise ValueError(f"Function `{name}` already exists in the list of functions")

        # Register component with dependency manager
        self._dependency_manager.register_component(name, ComponentGroup.FUNCTIONS)

        # Start building the function asynchronously using cleaner API
        self._dependency_manager.start_component_build(
            name, lambda: self._build_component_coordinated(name, ComponentGroup.FUNCTIONS, config))

        # Wait for it to complete and return the instance
        configured_function = await self._dependency_manager.wait_for_component(name, "add_function")
        return configured_function.instance

    @override
    def get_function(self, name: str | FunctionRef) -> Function:
        """Get a function synchronously - for backwards compatibility"""
        if name in self._functions:
            return self._functions[name].instance

        # Use dependency manager to check component state
        state = self._dependency_manager.get_component_state(name)
        if state == ComponentState.FAILED:
            # Re-raise the original exception for backward compatibility
            error = self._dependency_manager.get_component_error(name)
            raise error or RuntimeError(f"Function '{name}' failed to build")
        elif state == ComponentState.BUILDING:
            # Wait for the component using async patterns
            try:
                asyncio.get_running_loop()
                # We're in an async context, can't use asyncio.run()
                raise ValueError(f"Function `{name}` is still building. "
                                 f"Use await get_function_async() for better performance.")
            except RuntimeError:
                # We're not in an async context, can use asyncio.run()
                logger.debug("Function `%s` is building, waiting synchronously...", name)
                return asyncio.run(self.get_function_async(name, "get_function_sync"))

        # If component is defined in configuration, wait for it (normal building flow)
        if self._dependency_manager.is_component_defined(name):
            try:
                asyncio.get_running_loop()
                # We're in an async context, can't use asyncio.run()
                raise ValueError(f"Function `{name}` is not ready yet. "
                                 f"Use await get_function_async() for better performance.")
            except RuntimeError:
                # We're not in an async context, can use asyncio.run()
                logger.debug("Function `%s` not ready yet, waiting...", name)
                return asyncio.run(self.get_function_async(name, "get_function_sync"))

        # Only truly missing components get "not found" errors
        raise ValueError(f"Function `{name}` not found")

    async def get_function_async(self, name: str | FunctionRef, requester: str = "unknown") -> Function:
        """Get a function asynchronously, waiting if necessary"""
        configured_function = await self._dependency_manager.wait_for_component(name, requester)
        return configured_function.instance

    @override
    def get_function_config(self, name: str | FunctionRef) -> FunctionBaseConfig:
        if name not in self._functions:
            raise ValueError(f"Function `{name}` not found")
        return self._functions[name].config

    @override
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        """Set the workflow function"""
        if self._workflow is not None:
            warnings.warn("Overwriting existing workflow")

        # Clear previous workflow state using dependency manager
        self._dependency_manager.clear_component_state("<workflow>")
        await self._dependency_manager.cancel_build_task("<workflow>")

        # Register component with dependency manager
        self._dependency_manager.register_component("<workflow>", ComponentGroup.FUNCTIONS)

        # Start building the workflow asynchronously
        self._dependency_manager.start_component_build(
            "<workflow>", lambda: self._build_component_coordinated("<workflow>", ComponentGroup.FUNCTIONS, config))

        # Wait for it to complete
        configured_function = await self._dependency_manager.wait_for_component("<workflow>", "set_workflow")
        self._workflow = configured_function
        return configured_function.instance

    @override
    def get_workflow(self) -> Function:
        if self._workflow is None:
            raise ValueError("No workflow set")
        return self._workflow.instance

    @override
    def get_workflow_config(self) -> FunctionBaseConfig:
        if self._workflow is None:
            raise ValueError("No workflow set")
        return self._workflow.config

    @override
    def get_function_dependencies(self, fn_name: str | FunctionRef) -> FunctionDependencies:
        return self.function_dependencies[fn_name]

    @override
    def get_tool(self, fn_name: str | FunctionRef, wrapper_type: LLMFrameworkEnum | str):
        """Get a tool synchronously - for backwards compatibility"""
        # Check if function is already ready
        if fn_name in self._functions:
            fn = self._functions[fn_name]
            try:
                tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)
                return tool_wrapper_reg.build_fn(fn_name, fn.instance, self)
            except Exception as e:
                logger.error("Error fetching tool `%s`", fn_name, exc_info=True)
                raise e

        # Use dependency manager to check component state
        state = self._dependency_manager.get_component_state(fn_name)
        if state == ComponentState.FAILED:
            # Re-raise the original exception for backward compatibility
            error = self._dependency_manager.get_component_error(fn_name)
            raise error or RuntimeError(f"Function '{fn_name}' failed to build")
        elif state == ComponentState.BUILDING:
            # Wait for the component using async patterns
            try:
                asyncio.get_running_loop()
                # We're in an async context, can't use asyncio.run()
                raise ValueError(f"Function `{fn_name}` is still building. "
                                 f"Use await get_tool_async() for better performance.")
            except RuntimeError:
                # We're not in an async context, can use asyncio.run()
                logger.debug("Function `%s` is building, waiting synchronously...", fn_name)
                return asyncio.run(self.get_tool_async(fn_name, wrapper_type, "get_tool_sync"))

        # If component is defined in configuration, wait for it (normal building flow)
        if self._dependency_manager.is_component_defined(fn_name):
            try:
                asyncio.get_running_loop()
                # We're in an async context, can't use asyncio.run()
                raise ValueError(f"Function `{fn_name}` is not ready yet. "
                                 f"Use await get_tool_async() for better performance.")
            except RuntimeError:
                # We're not in an async context, can use asyncio.run()
                logger.debug("Function `%s` not ready yet, waiting...", fn_name)
                return asyncio.run(self.get_tool_async(fn_name, wrapper_type, "get_tool_sync"))

        # Only truly missing components get "not found" errors
        raise ValueError(f"Function `{fn_name}` not found")

    async def get_tool_async(self,
                             fn_name: str | FunctionRef,
                             wrapper_type: LLMFrameworkEnum | str,
                             requester: str = "unknown"):
        """Get a tool asynchronously, waiting if necessary"""
        # Wait for the function to be ready
        configured_function = await self._dependency_manager.wait_for_component(fn_name, requester)

        try:
            tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)
            return tool_wrapper_reg.build_fn(fn_name, configured_function.instance, self)
        except Exception as e:
            logger.error("Error fetching tool `%s`", fn_name, exc_info=True)
            raise e

    @override
    async def add_llm(self, name: str | LLMRef, config: LLMBaseConfig):
        """Add an LLM and start building it immediately"""
        if name in self._llms:
            raise ValueError(f"LLM `{name}` already exists in the list of LLMs")

        # Register component with dependency manager
        self._dependency_manager.register_component(name, ComponentGroup.LLMS)

        # Start building the LLM asynchronously
        self._dependency_manager.start_component_build(
            name, lambda: self._build_component_coordinated(name, ComponentGroup.LLMS, config))

        # Wait for it to complete
        await self._dependency_manager.wait_for_component(name, "add_llm")

    @override
    async def get_llm(self, llm_name: str | LLMRef, wrapper_type: LLMFrameworkEnum | str):
        """Get an LLM, waiting if it's not ready yet"""
        configured_llm = await self._dependency_manager.wait_for_component(llm_name, f"get_llm({wrapper_type})")

        try:
            # Generate wrapped client from registered client info
            client_info = self._registry.get_llm_client(config_type=type(configured_llm.config),
                                                        wrapper_type=wrapper_type)
            client = await self._get_exit_stack().enter_async_context(client_info.build_fn(configured_llm.config, self))
            return client
        except Exception as e:
            logger.error("Error getting llm `%s` with wrapper `%s`", llm_name, wrapper_type, exc_info=True)
            raise e

    @override
    def get_llm_config(self, llm_name: str | LLMRef) -> LLMBaseConfig:
        if llm_name not in self._llms:
            raise ValueError(f"LLM `{llm_name}` not found")
        return self._llms[llm_name].config

    @override
    async def add_embedder(self, name: str | EmbedderRef, config: EmbedderBaseConfig):
        """Add an embedder and start building it immediately"""
        if name in self._embedders:
            raise ValueError(f"Embedder `{name}` already exists in the list of embedders")

        # Register component with dependency manager
        self._dependency_manager.register_component(name, ComponentGroup.EMBEDDERS)

        # Start building the embedder asynchronously
        self._dependency_manager.start_component_build(
            name, lambda: self._build_component_coordinated(name, ComponentGroup.EMBEDDERS, config))

        # Wait for it to complete
        await self._dependency_manager.wait_for_component(name, "add_embedder")

    @override
    async def get_embedder(self, embedder_name: str | EmbedderRef, wrapper_type: LLMFrameworkEnum | str):
        """Get an embedder, waiting if it's not ready yet"""
        configured_embedder = await self._dependency_manager.wait_for_component(embedder_name,
                                                                                f"get_embedder({wrapper_type})")

        try:
            # Generate wrapped client from registered client info
            client_info = self._registry.get_embedder_client(config_type=type(configured_embedder.config),
                                                             wrapper_type=wrapper_type)
            client = await self._get_exit_stack().enter_async_context(
                client_info.build_fn(configured_embedder.config, self))
            return client
        except Exception as e:
            logger.error("Error getting embedder `%s` with wrapper `%s`", embedder_name, wrapper_type, exc_info=True)
            raise e

    @override
    def get_embedder_config(self, embedder_name: str | EmbedderRef) -> EmbedderBaseConfig:
        if embedder_name not in self._embedders:
            raise ValueError(f"Embedder `{embedder_name}` not found")
        return self._embedders[embedder_name].config

    @override
    async def add_memory_client(self, name: str | MemoryRef, config: MemoryBaseConfig) -> MemoryEditor:
        """Add a memory client and start building it immediately"""
        if name in self._memory_clients:
            raise ValueError(f"Memory `{name}` already exists in the list of memories")

        # Register component with dependency manager
        self._dependency_manager.register_component(name, ComponentGroup.MEMORY)

        # Start building the memory client asynchronously
        self._dependency_manager.start_component_build(
            name, lambda: self._build_component_coordinated(name, ComponentGroup.MEMORY, config))

        # Wait for it to complete and return the instance
        configured_memory = await self._dependency_manager.wait_for_component(name, "add_memory_client")
        return configured_memory.instance

    @override
    def get_memory_client(self, memory_name: str | MemoryRef) -> MemoryEditor:
        """Get a memory client synchronously - for backwards compatibility"""
        if memory_name in self._memory_clients:
            return self._memory_clients[memory_name].instance

        # Use dependency manager to check component state
        state = self._dependency_manager.get_component_state(memory_name)
        if state == ComponentState.FAILED:
            # Re-raise the original exception for backward compatibility
            error = self._dependency_manager.get_component_error(memory_name)
            raise error or RuntimeError(f"Memory '{memory_name}' failed to build")
        elif state == ComponentState.BUILDING:
            # Try to wait for the component using async patterns
            try:
                # Check if we're in an async context
                try:
                    asyncio.get_running_loop()
                    # We're in an async context, can't use asyncio.run()
                    raise ValueError(f"Memory `{memory_name}` is still building. "
                                     f"Use await get_memory_client_async() for better performance "
                                     f"in async context.")
                except RuntimeError:
                    # We're not in an async context, can use asyncio.run()
                    logger.debug("Memory `%s` is building, waiting synchronously...", memory_name)
                    return asyncio.run(self.get_memory_client_async(memory_name, "get_memory_client_sync"))
            except Exception as e:
                logger.error("Error waiting for memory `%s`", memory_name, exc_info=True)
                raise ValueError(f"Memory `{memory_name}` is still building. "
                                 f"Consider using await add_memory_client() or "
                                 f"get_memory_client_async() for better performance.") from e

        raise ValueError(f"Memory `{memory_name}` not found")

    async def get_memory_client_async(self, memory_name: str | MemoryRef, requester: str = "unknown") -> MemoryEditor:
        """Get a memory client asynchronously, waiting if necessary"""
        configured_memory = await self._dependency_manager.wait_for_component(memory_name, requester)
        return configured_memory.instance

    @override
    def get_memory_client_config(self, memory_name: str | MemoryRef) -> MemoryBaseConfig:
        if memory_name not in self._memory_clients:
            raise ValueError(f"Memory `{memory_name}` not found")
        return self._memory_clients[memory_name].config

    @override
    async def add_retriever(self, name: str | RetrieverRef, config: RetrieverBaseConfig):
        """Add a retriever and start building it immediately"""
        if name in self._retrievers:
            raise ValueError(f"Retriever '{name}' already exists in the list of retrievers")

        # Register component with dependency manager
        self._dependency_manager.register_component(name, ComponentGroup.RETRIEVERS)

        # Start building the retriever asynchronously
        self._dependency_manager.start_component_build(
            name, lambda: self._build_component_coordinated(name, ComponentGroup.RETRIEVERS, config))

        # Wait for it to complete
        await self._dependency_manager.wait_for_component(name, "add_retriever")

    @override
    async def get_retriever(self,
                            retriever_name: str | RetrieverRef,
                            wrapper_type: LLMFrameworkEnum | str | None = None):
        """Get a retriever, waiting if it's not ready yet"""
        configured_retriever = await self._dependency_manager.wait_for_component(retriever_name,
                                                                                 f"get_retriever({wrapper_type})")

        try:
            # Generate wrapped client from registered client info
            client_info = self._registry.get_retriever_client(config_type=type(configured_retriever.config),
                                                              wrapper_type=wrapper_type)
            client = await self._get_exit_stack().enter_async_context(
                client_info.build_fn(configured_retriever.config, self))
            return client
        except Exception as e:
            logger.error("Error getting retriever `%s` with wrapper `%s`", retriever_name, wrapper_type, exc_info=True)
            raise e

    @override
    async def get_retriever_config(self, retriever_name: str | RetrieverRef) -> RetrieverBaseConfig:
        if retriever_name not in self._retrievers:
            raise ValueError(f"Retriever `{retriever_name}` not found")
        return self._retrievers[retriever_name].config

    @override
    def get_user_manager(self) -> UserManagerHolder:
        return UserManagerHolder(context=AIQContext(self._context_state))

    async def add_logging_handler(self, name: str, config: LoggingBaseConfig):
        logging_info = self._registry.get_logging_method(type(config))
        handler = await self._get_exit_stack().enter_async_context(logging_info.build_fn(config, self))

        if not isinstance(handler, logging.Handler):
            raise TypeError(f"Expected a logging.Handler from {name}, got {type(handler)}")

        async with self._logging_handlers_lock:
            self._logging_handlers[name] = handler
            logging.getLogger().addHandler(handler)

    async def add_telemetry_exporter(self, name: str, config: TelemetryExporterBaseConfig) -> None:
        """Add a telemetry exporter to the builder"""
        exporter_info = self._registry.get_telemetry_exporter(type(config))

        # Build the exporter outside the lock (parallel)
        exporter_context_manager = exporter_info.build_fn(config, self)

        # Only protect the shared state modifications (serialized)
        async with self._telemetry_exporters_lock:
            exporter = await self._get_exit_stack().enter_async_context(exporter_context_manager)
            self._telemetry_exporters[name] = ConfiguredTelemetryExporter(config=config, instance=exporter)

    def build(self, entry_function: str | None = None) -> Workflow:
        """Build the workflow from the configured components"""
        if self._workflow is None:
            raise ValueError("Must set a workflow before building")

        # Build the config from the added objects
        config = AIQConfig(general=self.general_config,
                           functions={
                               k: v.config
                               for k, v in self._functions.items()
                           },
                           workflow=self._workflow.config,
                           llms={
                               k: v.config
                               for k, v in self._llms.items()
                           },
                           embedders={
                               k: v.config
                               for k, v in self._embedders.items()
                           },
                           memory={
                               k: v.config
                               for k, v in self._memory_clients.items()
                           },
                           retrievers={
                               k: v.config
                               for k, v in self._retrievers.items()
                           })

        if entry_function is None:
            entry_fn_obj = self.get_workflow()
        else:
            entry_fn_obj = self.get_function(entry_function)

        workflow = Workflow.from_entry_fn(config=config,
                                          entry_fn=entry_fn_obj,
                                          functions={
                                              k: v.instance
                                              for k, v in self._functions.items()
                                          },
                                          llms={
                                              k: v.instance
                                              for k, v in self._llms.items()
                                          },
                                          embeddings={
                                              k: v.instance
                                              for k, v in self._embedders.items()
                                          },
                                          memory={
                                              k: v.instance
                                              for k, v in self._memory_clients.items()
                                          },
                                          telemetry_exporters={
                                              k: v.instance
                                              for k, v in self._telemetry_exporters.items()
                                          },
                                          retrievers={
                                              k: v.instance
                                              for k, v in self._retrievers.items()
                                          },
                                          context_state=self._context_state)

        return workflow

    async def populate_builder(self, config: AIQConfig, skip_workflow: bool = False):
        """
        Populate the builder with all components from the configuration.
        All components start building immediately and resolve dependencies on-demand.
        """
        # Count components for logging
        llm_count = len(config.llms)
        embedder_count = len(config.embedders)
        memory_count = len(config.memory)
        retriever_count = len(config.retrievers)
        function_count = len(config.functions)
        tracing_count = len(config.general.telemetry.tracing)
        workflow_count = 0 if skip_workflow else 1
        total_count = (llm_count + embedder_count + memory_count + retriever_count + function_count + tracing_count +
                       workflow_count)

        logger.info(
            "Discovered %d components: %d LLMs, %d embedders, %d memory, %d retrievers, "
            "%d functions, %d tracing exporters%s",
            total_count,
            llm_count,
            embedder_count,
            memory_count,
            retriever_count,
            function_count,
            tracing_count,
            ", 1 workflow" if not skip_workflow else "")

        # Register all components with dependency manager
        for name in config.llms.keys():
            self._dependency_manager.register_component(name, ComponentGroup.LLMS)
        for name in config.embedders.keys():
            self._dependency_manager.register_component(name, ComponentGroup.EMBEDDERS)
        for name in config.memory.keys():
            self._dependency_manager.register_component(name, ComponentGroup.MEMORY)
        for name in config.retrievers.keys():
            self._dependency_manager.register_component(name, ComponentGroup.RETRIEVERS)
        for name in config.functions.keys():
            self._dependency_manager.register_component(name, ComponentGroup.FUNCTIONS)
        # Register exporters for parallel building
        for name in config.general.telemetry.tracing.keys():
            self._dependency_manager.register_component(name, ComponentGroup.TRACING)
        if not skip_workflow:
            self._dependency_manager.register_component("<workflow>", ComponentGroup.FUNCTIONS)

        # Start all components building in parallel
        build_tasks = []

        # Start LLMs
        for name, llm_config in config.llms.items():
            task = self._dependency_manager.start_component_build(
                name, lambda cfg=llm_config, n=name: self._build_component_coordinated(n, ComponentGroup.LLMS, cfg))
            build_tasks.append(task)

        # Start embedders
        for name, embedder_config in config.embedders.items():
            task = self._dependency_manager.start_component_build(
                name,
                lambda cfg=embedder_config, n=name: self._build_component_coordinated(n, ComponentGroup.EMBEDDERS, cfg))
            build_tasks.append(task)

        # Start memory clients
        for name, memory_config in config.memory.items():
            task = self._dependency_manager.start_component_build(
                name,
                lambda cfg=memory_config, n=name: self._build_component_coordinated(n, ComponentGroup.MEMORY, cfg))
            build_tasks.append(task)

        # Start retrievers
        for name, retriever_config in config.retrievers.items():
            task = self._dependency_manager.start_component_build(
                name, lambda cfg=retriever_config, n=name: self._build_component_coordinated(
                    n, ComponentGroup.RETRIEVERS, cfg))
            build_tasks.append(task)

        # Start functions
        for name, function_config in config.functions.items():
            task = self._dependency_manager.start_component_build(
                name,
                lambda cfg=function_config, n=name: self._build_component_coordinated(n, ComponentGroup.FUNCTIONS, cfg))
            build_tasks.append(task)

        # Start exporters
        for name, exporter_config in config.general.telemetry.tracing.items():
            task = self._dependency_manager.start_component_build(
                name,
                lambda cfg=exporter_config, n=name: self._build_component_coordinated(n, ComponentGroup.TRACING, cfg))
            build_tasks.append(task)

        # Start workflow if requested
        if not skip_workflow:
            task = self._dependency_manager.start_component_build(
                "<workflow>",
                lambda: self._build_component_coordinated("<workflow>", ComponentGroup.FUNCTIONS, config.workflow))
            build_tasks.append(task)

        # Wait for all components to be built
        logger.info("Initializing %d components...", len(build_tasks))
        results = await asyncio.gather(*build_tasks, return_exceptions=True)

        # Check for any failures
        failures = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append(result)

        if failures:
            logger.error("Failed to build %d components:", len(failures))
            for error in failures:
                logger.error("  - %s", error)
            raise RuntimeError(f"Failed to build {len(failures)} components")

        logger.info("All components built successfully")

    @classmethod
    @asynccontextmanager
    async def from_config(cls, config: AIQConfig):
        """Create a builder from a configuration"""
        async with cls(general_config=config.general) as builder:
            await builder.populate_builder(config)
            yield builder


class AsyncChildBuilder(Builder):
    """
    A fully async child builder that provides dynamic dependency resolution for component build functions.
    This builder is passed to component build functions and uses await patterns for all dependencies.

    Now uses the parent's DependencyManager for coordination.
    """

    def __init__(self, parent_builder: WorkflowBuilder, requester_name: str = "unknown"):
        self._parent_builder = parent_builder
        self._requester_name = requester_name
        self._dependencies = FunctionDependencies()

    @property
    def dependencies(self) -> FunctionDependencies:
        return self._dependencies

    @override
    async def add_function(self, name: str, config: FunctionBaseConfig) -> Function:
        return await self._parent_builder.add_function(name, config)

    @override
    def get_function(self, name: str) -> Function:
        """Synchronous access for backwards compatibility"""
        if name in self._parent_builder._functions:
            return self._parent_builder._functions[name].instance

        # Use dependency manager to check if component is defined
        if self._parent_builder._dependency_manager.is_component_defined(name):
            logger.debug("Component %s waiting for function %s", self._requester_name, name)
            from aiq.builder.exceptions import DependencyNotReadyError
            raise DependencyNotReadyError(f"Function `{name}` is not ready yet", name)

        raise ValueError(f"Function `{name}` not found")

    async def get_function_async(self, name: str) -> Function:
        """Async access - preferred for new code"""
        function = await self._parent_builder._dependency_manager.wait_for_component(name, self._requester_name)
        self._dependencies.add_function(name)
        return function.instance

    @override
    def get_function_config(self, name: str) -> FunctionBaseConfig:
        return self._parent_builder.get_function_config(name)

    @override
    async def set_workflow(self, config: FunctionBaseConfig) -> Function:
        return await self._parent_builder.set_workflow(config)

    @override
    def get_workflow(self) -> Function:
        return self._parent_builder.get_workflow()

    @override
    def get_workflow_config(self) -> FunctionBaseConfig:
        return self._parent_builder.get_workflow_config()

    @override
    def get_tools(self, tool_names: list[str], wrapper_type: LLMFrameworkEnum | str) -> list[Any]:
        """Get tools with graceful dependency waiting for normal building flow"""
        tools = []
        for fn_name in tool_names:
            # Check if function is already ready
            if fn_name in self._parent_builder._functions:
                fn = self._parent_builder._functions[fn_name]
                try:
                    tool_wrapper_reg = self._parent_builder._registry.get_tool_wrapper(llm_framework=wrapper_type)
                    tool = tool_wrapper_reg.build_fn(fn_name, fn.instance, self._parent_builder)
                    tools.append(tool)
                    continue
                except Exception as e:
                    logger.error("Error fetching tool `%s`", fn_name, exc_info=True)
                    raise e

            # If component is defined in configuration, this is normal building flow
            if self._parent_builder._dependency_manager.is_component_defined(fn_name):
                # This is normal during parallel building - wait briefly and retry
                logger.debug("Component %s waiting for function %s", self._requester_name, fn_name)

                # Since we're in an async context, we need to handle this gracefully
                # Create a special exception that indicates dependency waiting (not an error)
                from aiq.builder.exceptions import DependencyNotReadyError
                raise DependencyNotReadyError(f"Function `{fn_name}` is not ready yet", fn_name)

            # Only truly missing components get "not found" errors
            raise ValueError(f"Function `{fn_name}` not found")

        return tools

    @override
    def get_tool(self, fn_name: str, wrapper_type: LLMFrameworkEnum | str):
        """Synchronous access for backwards compatibility"""
        # During building, we should be able to wait for dependencies gracefully
        if fn_name in self._parent_builder._functions:
            fn = self._parent_builder._functions[fn_name]
            try:
                tool_wrapper_reg = self._parent_builder._registry.get_tool_wrapper(llm_framework=wrapper_type)
                return tool_wrapper_reg.build_fn(fn_name, fn.instance, self._parent_builder)
            except Exception as e:
                logger.error("Error fetching tool `%s`", fn_name, exc_info=True)
                raise e

        # If component is defined in configuration, this is normal building flow
        if self._parent_builder._dependency_manager.is_component_defined(fn_name):
            # This is normal during parallel building - signal dependency waiting
            logger.debug("Component %s waiting for function %s", self._requester_name, fn_name)
            from aiq.builder.exceptions import DependencyNotReadyError
            raise DependencyNotReadyError(f"Function `{fn_name}` is not ready yet", fn_name)

        # Only truly missing components get "not found" errors
        raise ValueError(f"Function `{fn_name}` not found")

    async def get_tool_async(self, fn_name: str, wrapper_type: LLMFrameworkEnum | str):
        """Async access - preferred for new code"""
        tool = await self._parent_builder.get_tool_async(fn_name, wrapper_type, self._requester_name)
        self._dependencies.add_function(fn_name)
        return tool

    @override
    async def add_llm(self, name: str, config: LLMBaseConfig):
        return await self._parent_builder.add_llm(name, config)

    @override
    async def get_llm(self, llm_name: str, wrapper_type: LLMFrameworkEnum | str):
        llm = await self._parent_builder.get_llm(llm_name, wrapper_type)
        self._dependencies.add_llm(llm_name)
        return llm

    @override
    def get_llm_config(self, llm_name: str) -> LLMBaseConfig:
        return self._parent_builder.get_llm_config(llm_name)

    @override
    async def add_embedder(self, name: str, config: EmbedderBaseConfig):
        return await self._parent_builder.add_embedder(name, config)

    @override
    async def get_embedder(self, embedder_name: str, wrapper_type: LLMFrameworkEnum | str):
        embedder = await self._parent_builder.get_embedder(embedder_name, wrapper_type)
        self._dependencies.add_embedder(embedder_name)
        return embedder

    @override
    def get_embedder_config(self, embedder_name: str) -> EmbedderBaseConfig:
        return self._parent_builder.get_embedder_config(embedder_name)

    @override
    async def add_memory_client(self, name: str, config: MemoryBaseConfig) -> MemoryEditor:
        return await self._parent_builder.add_memory_client(name, config)

    @override
    def get_memory_client(self, memory_name: str) -> MemoryEditor:
        """Synchronous access for backwards compatibility"""
        return self._parent_builder.get_memory_client(memory_name)

    async def get_memory_client_async(self, memory_name: str) -> MemoryEditor:
        """Async access - preferred for new code"""
        memory_client = await self._parent_builder.get_memory_client_async(memory_name, self._requester_name)
        self._dependencies.add_memory_client(memory_name)
        return memory_client

    @override
    def get_memory_client_config(self, memory_name: str) -> MemoryBaseConfig:
        return self._parent_builder.get_memory_client_config(memory_name)

    @override
    async def add_retriever(self, name: str, config: RetrieverBaseConfig):
        return await self._parent_builder.add_retriever(name, config)

    @override
    async def get_retriever(self, retriever_name: str, wrapper_type: LLMFrameworkEnum | str | None = None):
        retriever = await self._parent_builder.get_retriever(retriever_name, wrapper_type)
        self._dependencies.add_retriever(retriever_name)
        return retriever

    @override
    async def get_retriever_config(self, retriever_name: str) -> RetrieverBaseConfig:
        return await self._parent_builder.get_retriever_config(retriever_name)

    @override
    def get_user_manager(self) -> UserManagerHolder:
        return self._parent_builder.get_user_manager()

    @override
    def get_function_dependencies(self, fn_name: str) -> FunctionDependencies:
        return self._parent_builder.get_function_dependencies(fn_name)
