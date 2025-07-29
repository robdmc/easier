"""
AgentRunner module for processing prompts with EZAgent in batches with concurrency control.

This module provides the AgentRunner class that can process multiple prompts
concurrently using an EZAgent, with optional database persistence using DuckDB.
"""

import asyncio
import atexit
import os
import signal
import threading
import weakref
from typing import Callable, Any, Generator, Union, List, Optional, TYPE_CHECKING, Set

if TYPE_CHECKING:
    import easier  # type: ignore
    import pandas as pd
    import pydantic_ai.agent
    import duckdb


class TaskTracker:
    """Global singleton to track all running asyncio tasks across AgentRunner instances"""
    _instance: Optional["TaskTracker"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "TaskTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.active_runners: weakref.WeakSet["AgentRunner"] = weakref.WeakSet()
        self.active_tasks: Set[asyncio.Task] = set()
        self._cleanup_registered = False
        self._initialized = True
        
        # Register cleanup handlers
        if not self._cleanup_registered:
            atexit.register(self.cleanup_all)
            self._cleanup_registered = True
    
    def register_runner(self, runner: "AgentRunner") -> None:
        """Register an AgentRunner instance"""
        self.active_runners.add(runner)
    
    def add_task(self, task: asyncio.Task) -> None:
        """Add a task to track"""
        self.active_tasks.add(task)
        
        # Clean up completed tasks
        def cleanup_task(t: asyncio.Task) -> None:
            self.active_tasks.discard(t)
        
        task.add_done_callback(cleanup_task)
    
    def cancel_all_tasks(self) -> None:
        """Cancel all tracked tasks"""
        print(f"Cancelling {len(self.active_tasks)} active tasks...")
        
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
        
        # Clear the set
        self.active_tasks.clear()
    
    def cleanup_all(self) -> None:
        """Cleanup all runners and tasks - called by atexit"""
        try:
            print("TaskTracker: Performing cleanup on exit...")
            
            # Cancel all tasks
            self.cancel_all_tasks()
            
            # Cleanup all runners
            for runner in list(self.active_runners):
                try:
                    runner._force_cleanup()
                except Exception as e:
                    print(f"Error during runner cleanup: {e}")
                    
        except Exception as e:
            print(f"Error during TaskTracker cleanup: {e}")


# Global task tracker instance
_task_tracker = TaskTracker()


class DatabaseSchemaValidationError(Exception):
    """Raised when DataFrame data cannot be converted to the expected database schema"""
    pass


class AgentRunner:
    """
    Concurrent batch processor for EZAgent prompts with optional database persistence.
    
    Processes multiple prompts using an EZAgent with built-in concurrency control,
    timeout handling, and optional DuckDB persistence for high-throughput AI workflows.
    
    Key features:
    - Concurrent processing with configurable limits
    - Automatic batch management and timeout protection  
    - Optional database persistence and graceful shutdown
    - Context manager support for resource cleanup
    
    Basic usage:
        with ezr.AgentRunner(agent) as runner:
            results = await runner.run(prompts, batch_size=10, max_concurrency=5)
    """
    def __init__(
        self,
        agent: "easier.EZAgent",  # type: ignore
        db_file: Optional[str] = None,
        overwrite: bool = False,
        table_name: str = "results",
        timeout: float = 300.0,  # 5 minute default timeout
    ) -> None:
        """
        Initialize AgentRunner for batch processing with optional database persistence.
        
        Args:
            agent: EZAgent instance for processing prompts
            db_file: Path to DuckDB database file for result persistence (optional)
            overwrite: If True, delete existing database file (default False)
            table_name: Database table name for results (default "results") 
            timeout: Max seconds per prompt before timeout (default 300.0)
            
        Example:
            with ezr.AgentRunner(agent, db_file="results.db", timeout=60.0) as runner:
                results = await runner.run(prompts)
            
        Note: Database persistence requires framer_func in run() method.
        """
        import easier as ezr

        if not isinstance(agent, ezr.EZAgent):
            raise TypeError(
                f"Expected agent to be of type ezr.EZAgent, got {type(agent)}"
            )
        self.agent: "easier.EZAgent" = agent

        # Database configuration
        self.db_enabled: bool = db_file is not None
        self.db_file: Optional[str] = db_file
        self.overwrite: bool = overwrite
        self.table_name: str = table_name
        
        # Timeout configuration
        self.timeout: float = timeout
        
        # Task management
        self.active_tasks: Set[asyncio.Task] = set()
        self._is_running: bool = False
        self._cleanup_done: bool = False
        self._in_context: bool = False
        
        # Usage tracking
        import pandas as pd
        self._usage_stats: "pd.Series" = pd.Series({
            "requests": 0,
            "request_tokens": 0,
            "response_tokens": 0,
            "thoughts_tokens": 0,
            "total_tokens": 0
        })
        self._usage_lock: threading.Lock = threading.Lock()
        
        # Register with global task tracker
        _task_tracker.register_runner(self)

        # Initialize database if enabled
        if self.db_enabled:
            self._init_database()
    
    def _map_python_type_to_duckdb(self, python_type) -> str:
        """Map Python/Pydantic types to DuckDB column types"""
        import typing
        from enum import Enum
        
        # Handle None type
        if python_type is type(None):
            return "VARCHAR"
        
        # Basic types
        if python_type is int:
            return "INTEGER"
        elif python_type is float:
            return "DOUBLE"
        elif python_type is bool:
            return "BOOLEAN"
        elif python_type is str:
            return "VARCHAR"
        elif python_type is bytes:
            return "BLOB"
        
        # Handle generic types (List, Dict, Optional, Union, etc.)
        if hasattr(python_type, '__origin__'):
            origin = python_type.__origin__
            
            if origin is list:
                # Lists become JSON arrays
                return "JSON"
            elif origin is dict:
                # Dicts become JSON objects
                return "JSON"
            elif origin is typing.Union:
                # Handle Optional[T] and Union types
                args = python_type.__args__
                if len(args) == 2 and type(None) in args:
                    # This is Optional[T], get the non-None type
                    non_none_type = next(arg for arg in args if arg is not type(None))
                    return self._map_python_type_to_duckdb(non_none_type)
                else:
                    # Complex union - store as JSON to preserve flexibility
                    return "JSON"
            elif origin is tuple:
                # Tuples become JSON arrays
                return "JSON"
        
        # Handle Enum types
        if hasattr(python_type, '__bases__') and any(issubclass(base, Enum) for base in python_type.__bases__ if isinstance(base, type)):
            return "VARCHAR"
        
        # Handle Pydantic models (have model_fields attribute)
        if hasattr(python_type, 'model_fields'):
            return "JSON"
        
        # Handle datetime types
        try:
            import datetime
            if python_type in (datetime.datetime, datetime.date, datetime.time):
                return "TIMESTAMP"
        except ImportError:
            pass
        
        # Default: complex or unknown types become JSON
        print(f"DEBUG: Unknown type {python_type}, defaulting to JSON")
        return "JSON"
    
    def __enter__(self) -> "AgentRunner":
        """Context manager entry"""
        self._in_context = True
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup"""
        self._in_context = False
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Cancel all tasks and cleanup resources"""
        if self._cleanup_done:
            return
            
        print(f"AgentRunner cleanup: Cancelling {len(self.active_tasks)} tasks...")
        
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
        
        self.active_tasks.clear()
        self._is_running = False
        self._cleanup_done = True
    
    def _force_cleanup(self) -> None:
        """Force cleanup - called by TaskTracker"""
        self._cleanup()

    def _init_database(self) -> None:
        """Initialize the DuckDB database and table if enabled"""
        if self.overwrite and self.db_file is not None:
            try:
                os.remove(self.db_file)
                print(f"Deleted existing {self.db_file} for overwrite.")
            except FileNotFoundError:
                pass

        # Connect and create table structure based on first framed result
        # Since we don't know the schema yet, we'll defer table creation to first write

    def _create_batches(
        self, prompts: List[str], batch_size: int
    ) -> Generator[List[str], None, None]:
        """Generator that yields batches of prompts"""
        for i in range(0, len(prompts), batch_size):
            yield prompts[i : i + batch_size]

    async def _process_batch(
        self,
        batch_prompts: List[str],
        db_write_semaphore: Optional["asyncio.Semaphore"] = None,
        framer_func: Optional[
            Callable[
                [List[Optional["pydantic_ai.agent.AgentRunResult"]]], "pd.DataFrame"
            ]
        ] = None,
        output_type: Any = None,
    ) -> Union["pd.DataFrame", List[Optional["pydantic_ai.agent.AgentRunResult"]]]:
        """Process a batch of prompts with timeout support"""
        try:
            results: List[Optional["pydantic_ai.agent.AgentRunResult"]] = []
            for prompt in batch_prompts:
                try:
                    # Add timeout wrapper for each LLM call
                    result: "pydantic_ai.agent.AgentRunResult" = await asyncio.wait_for(
                        self.agent.run(prompt, output_type=output_type),
                        timeout=self.timeout
                    )
                    results.append(result)
                    
                    # Aggregate usage stats in a thread-safe manner
                    try:
                        usage_series = self.agent.get_usage(result)
                        with self._usage_lock:
                            self._usage_stats += usage_series
                    except Exception as usage_error:
                        print(f"Failed to aggregate usage stats: {usage_error}")
                except asyncio.TimeoutError:
                    print(f"Prompt timed out after {self.timeout}s: {prompt[:100]}...")
                    results.append(None)
                except Exception as prompt_error:
                    print(f"Prompt failed: {prompt_error}")
                    results.append(None)

            # Apply framer_func if provided
            if framer_func is not None:
                framed_results: "pd.DataFrame" = framer_func(results)

                # Write to database if enabled
                if self.db_enabled and db_write_semaphore is not None:
                    await self._write_batch_to_db(framed_results, db_write_semaphore, output_type)

                return framed_results
            else:
                return results
        except DatabaseSchemaValidationError:
            # Re-raise schema validation errors immediately to stop all processing
            raise
        except Exception as e:
            print(f"Batch processing failed: {e}")
            raise

    async def _write_batch_to_db(
        self, df: "pd.DataFrame", db_write_semaphore: "asyncio.Semaphore", output_type: Any = None
    ) -> None:
        """Write batch results to DuckDB"""
        import duckdb

        async with db_write_semaphore:
            try:
                # DEBUG: Print dataframe info before writing
                print(f"DEBUG: Writing batch to DB with {len(df)} rows")
                print(f"DEBUG: DataFrame columns: {list(df.columns)}")
                print(f"DEBUG: DataFrame dtypes: {df.dtypes.to_dict()}")
                if len(df) > 0:
                    print(f"DEBUG: First row data: {df.iloc[0].to_dict()}")

                # Create database connection for this batch
                # self.db_file is guaranteed to be not None when db_enabled is True
                assert self.db_file is not None  # Help mypy understand this is not None
                con: "duckdb.DuckDBPyConnection" = duckdb.connect(self.db_file)

                try:
                    # Check if table exists
                    table_exists_result = con.execute(f"""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_name = '{self.table_name}'
                    """).fetchone()
                    table_exists = table_exists_result[0] > 0 if table_exists_result else False

                    if not table_exists and output_type is not None and not df.empty:
                        # Use Pydantic model as source of truth for table schema
                        print(f"DEBUG: Creating table '{self.table_name}' using Pydantic schema")
                        
                        schema_parts = []
                        model_fields = output_type.model_fields if hasattr(output_type, 'model_fields') else {}
                        
                        # Create schema based ONLY on Pydantic model fields
                        for field_name, field_info in model_fields.items():
                            # Map Pydantic types to DuckDB types
                            field_type = field_info.annotation if hasattr(field_info, 'annotation') else str
                            print(f"DEBUG: Field {field_name} has type {field_type}")
                            
                            db_type = self._map_python_type_to_duckdb(field_type)
                            schema_parts.append(f'"{field_name}" {db_type}')
                        
                        if schema_parts:
                            create_sql = f"CREATE TABLE {self.table_name} ({', '.join(schema_parts)})"
                            print(f"DEBUG: Table creation SQL: {create_sql}")
                            con.execute(create_sql)
                        else:
                            # Fallback if we can't get schema
                            print("DEBUG: No schema found, creating table via SELECT")
                            con.execute(f"CREATE TABLE {self.table_name} AS SELECT * FROM df WHERE FALSE")
                    
                    elif not table_exists and output_type is not None and df.empty:
                        # Handle empty DataFrame with Pydantic schema - create table from Pydantic model only
                        print(f"DEBUG: Creating table '{self.table_name}' using Pydantic schema")
                        
                        schema_parts = []
                        model_fields = output_type.model_fields if hasattr(output_type, 'model_fields') else {}
                        
                        for field_name, field_info in model_fields.items():
                            # Map Pydantic types to DuckDB types
                            field_type = field_info.annotation if hasattr(field_info, 'annotation') else str
                            print(f"DEBUG: Field {field_name} has type {field_type}")
                            
                            db_type = self._map_python_type_to_duckdb(field_type)
                            schema_parts.append(f'"{field_name}" {db_type}')
                        
                        if schema_parts:
                            create_sql = f"CREATE TABLE {self.table_name} ({', '.join(schema_parts)})"
                            print(f"DEBUG: Table creation SQL: {create_sql}")
                            con.execute(create_sql)
                    
                    elif not table_exists and not df.empty:
                        # Fallback if no output_type provided but we have data
                        print(f"DEBUG: Creating table '{self.table_name}' via SELECT fallback")
                        con.execute(f"CREATE TABLE {self.table_name} AS SELECT * FROM df WHERE FALSE")
                    elif not table_exists and df.empty:
                        # Can't create table from empty DataFrame with no schema info - skip
                        print("DEBUG: Skipping table creation for empty DataFrame with no schema")
                        return

                    # Insert the data if DataFrame is not empty
                    if not df.empty:
                        try:
                            # Filter DataFrame to only include Pydantic model fields
                            import json
                            
                            if output_type is not None and hasattr(output_type, 'model_fields'):
                                # Only keep columns that are in the Pydantic model
                                model_columns = list(output_type.model_fields.keys())
                                available_columns = [col for col in model_columns if col in df.columns]
                                df_to_insert = df[available_columns].copy()
                                print(f"DEBUG: Filtered DataFrame from {list(df.columns)} to {available_columns}")
                                
                                # Serialize fields that should be JSON (lists, dicts)
                                for field_name, field_info in output_type.model_fields.items():
                                    if field_name in df_to_insert.columns:
                                        field_type = field_info.annotation if hasattr(field_info, 'annotation') else str
                                        # Check if this field should be JSON (lists, dicts)
                                        if hasattr(field_type, '__origin__'):
                                            origin = field_type.__origin__
                                            if origin in (list, dict):
                                                # Serialize to JSON strings
                                                df_to_insert[field_name] = df_to_insert[field_name].apply(
                                                    lambda x: json.dumps(x) if x is not None else None
                                                )
                            else:
                                # No output_type schema, use all columns
                                df_to_insert = df.copy()
                            
                            con.execute(f"INSERT INTO {self.table_name} SELECT * FROM df_to_insert")
                        except Exception as conversion_error:
                            # Get table schema for debugging
                            try:
                                schema_result = con.execute(f"DESCRIBE {self.table_name}").fetchall()
                                table_schema = {row[0]: row[1] for row in schema_result}  # column_name: column_type
                            except Exception:
                                table_schema = "Unable to retrieve table schema"
                            
                            # Get DataFrame dtypes for debugging
                            df_dtypes = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
                            
                            # Create detailed error message
                            error_msg = (
                                f"Database type conversion failed for table '{self.table_name}':\n"
                                f"Original error: {conversion_error}\n"
                                f"Table schema: {table_schema}\n"
                                f"DataFrame dtypes: {df_dtypes}\n"
                                f"DataFrame sample (first row): {df.iloc[0].to_dict() if len(df) > 0 else 'Empty DataFrame'}"
                            )
                            
                            # Re-raise as our custom exception to trigger job cancellation
                            raise DatabaseSchemaValidationError(error_msg) from conversion_error

                except Exception as db_error:
                    try:
                        con.rollback()
                    except Exception:
                        pass
                    raise db_error
                finally:
                    con.close()

            except Exception as e:
                print(f"Database write failed: {e}")
                raise

    async def run(
        self,
        prompts: List[str],
        batch_size: int = 10,
        max_concurrency: int = 10,
        framer_func: Optional[
            Callable[
                [List[Optional["pydantic_ai.agent.AgentRunResult"]]], "pd.DataFrame"
            ]
        ] = None,
        output_type: Any = None,
    ) -> Union["pd.DataFrame", List[Optional["pydantic_ai.agent.AgentRunResult"]]]:
        """
        Process multiple prompts concurrently with batch management and error handling.
        
        Args:
            prompts: List of prompt strings to process
            batch_size: Number of prompts per batch (default 10, recommend 10-50)
            max_concurrency: Max simultaneous batches (default 10, recommend 5-20)
            framer_func: Function to convert results to DataFrame (required for DB persistence)
            output_type: Expected output type for structured responses (optional)
            
        Returns:
            DataFrame if framer_func provided, otherwise List of AgentRunResults.
            Failed prompts return None in results.
            
        Example:
            with ezr.AgentRunner(agent) as runner:
                results = await runner.run(
                    prompts, 
                    batch_size=20, 
                    max_concurrency=5,
                    framer_func=my_framer
                )
                
        Note: Handles timeouts, interrupts, and API failures gracefully.
        Prints progress and saves to database if configured.
        """
        
        # Check if running in context manager
        if not self._in_context:
            raise RuntimeError(
                "AgentRunner.run() must be called within a context manager. "
                "Use 'with AgentRunner(agent) as runner:' before calling run()."
            )

        # Validate that if database is enabled, framer_func is provided
        if self.db_enabled and framer_func is None:
            raise ValueError(
                f"Database file '{self.db_file}' is configured but framer_func is not provided. "
                "Both db_file and framer_func are required for database persistence."
            )

        # Set up signal handler for graceful shutdown
        shutdown_event: asyncio.Event = asyncio.Event()
        
        def signal_handler(signum: int, frame: Any) -> None:
            print(f"Received signal {signum}, initiating immediate shutdown...")
            shutdown_event.set()
            # Cancel all tasks immediately
            _task_tracker.cancel_all_tasks()
            self._cleanup()

        # Register signal handlers
        original_sigint: Any = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm: Any = signal.signal(signal.SIGTERM, signal_handler)
        
        # Mark as running
        self._is_running = True

        try:
            # Split prompts into batches
            batches: List[List[str]] = list(self._create_batches(prompts, batch_size))
            total_batches: int = len(batches)
            semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrency)
            db_write_semaphore: Optional[asyncio.Semaphore] = (
                asyncio.Semaphore(1) if self.db_enabled else None
            )

            print(
                f"Processing {len(prompts)} prompts in {total_batches} batches (batch_size={batch_size}, max_concurrency={max_concurrency})"
            )

            async def process_batch_with_semaphore(
                batch: List[str], batch_idx: int
            ) -> Optional[
                Union[
                    "pd.DataFrame", List[Optional["pydantic_ai.agent.AgentRunResult"]]
                ]
            ]:
                try:
                    async with semaphore:
                        print(f"Starting batch {batch_idx+1} of {total_batches}")
                        result: Union[
                            "pd.DataFrame",
                            List[Optional["pydantic_ai.agent.AgentRunResult"]],
                        ] = await self._process_batch(
                            batch, db_write_semaphore, framer_func, output_type
                        )
                        print(f"Finished batch {batch_idx+1} of {total_batches}")
                        return result
                except DatabaseSchemaValidationError as e:
                    # Re-raise schema validation errors as they should fail fast
                    print(f"Failed to process batch {batch_idx+1}: {e}")
                    raise
                except Exception as e:
                    print(f"Failed to process batch {batch_idx+1}: {e}")
                    return None

            # Create tasks for all batches
            batch_tasks: List[
                "asyncio.Task[Optional[Union[pd.DataFrame, List[Optional[pydantic_ai.agent.AgentRunResult]]]]]"
            ] = []
            
            for idx, batch in enumerate(batches):
                task = asyncio.create_task(process_batch_with_semaphore(batch, idx))
                batch_tasks.append(task)
                
                # Track tasks in both local and global trackers
                self.active_tasks.add(task)
                _task_tracker.add_task(task)
                
                # Remove from local tracker when done
                def cleanup_local_task(t: asyncio.Task, runner_ref: "AgentRunner" = self) -> None:
                    runner_ref.active_tasks.discard(t)
                
                task.add_done_callback(cleanup_local_task)

            try:
                # Wait for all batches to complete
                results: List[Any] = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
                
                # Check for DatabaseSchemaValidationError and re-raise it
                for result in results:
                    if isinstance(result, DatabaseSchemaValidationError):
                        raise result
            except KeyboardInterrupt:
                print("Received interrupt signal, cancelling all tasks...")

                # Cancel all running tasks
                for task in batch_tasks:
                    if not task.done():
                        task.cancel()

                # Wait for tasks to be cancelled with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    print("Some tasks did not cancel gracefully within timeout")

                # Count completed batches
                completed_results: List[Any] = [
                    r for r in batch_tasks if r.done() and not r.cancelled()
                ]
                print(
                    f"Processing interrupted. Completed {len(completed_results)}/{total_batches} batches."
                )

                # For interrupted execution, return flattened list
                interrupted_results: List[
                    Optional["pydantic_ai.agent.AgentRunResult"]
                ] = []
                for r in completed_results:
                    if not isinstance(r.result(), Exception) and r.result() is not None:
                        if isinstance(r.result(), list):
                            interrupted_results.extend(r.result())
                        else:
                            # If it's a DataFrame, we can't flatten it properly here
                            # This is a limitation of the interrupted execution path
                            pass
                return interrupted_results

            # Count successful and failed batches
            successful_batches: int = sum(
                1
                for result in results
                if not isinstance(result, Exception) and result is not None
            )
            failed_batches: int = sum(
                1
                for result in results
                if isinstance(result, Exception) or result is None
            )

            print(
                f"Processed {successful_batches}/{total_batches} batches successfully. {failed_batches} batches failed."
            )

            # Flatten results (results are already framed if framer_func was provided)
            if framer_func is not None:
                # Results are DataFrames
                dataframe_results: List["pd.DataFrame"] = []
                for result in results:
                    if result is not None and not isinstance(result, Exception):
                        dataframe_results.append(result)

                # Concatenate DataFrames
                if dataframe_results:
                    import pandas as pd

                    return pd.concat(dataframe_results, ignore_index=True)
                else:
                    import pandas as pd

                    return pd.DataFrame()  # Return empty DataFrame
            else:
                # Results are lists of AgentRunResult
                flattened_results: List[
                    Optional["pydantic_ai.agent.AgentRunResult"]
                ] = []
                for result in results:
                    if result is not None and not isinstance(result, Exception):
                        flattened_results.extend(result)
                return flattened_results

        finally:
            # Ensure cleanup
            self._is_running = False
            self._cleanup()
            
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def get_usage(self, input_ppm_cost: float = 1, output_ppm_cost: float = 1, thought_ppm_cost: float = 1) -> "pd.Series":
        """
        Get aggregated usage statistics with cost calculations.
        
        Args:
            input_ppm_cost: Cost per million input tokens (default 1)
            output_ppm_cost: Cost per million output tokens (default 1)
            thought_ppm_cost: Cost per million thought tokens (default 1)
            
        Returns:
            A pandas Series containing usage stats and calculated costs:
            - 'requests': Total number of API requests made
            - 'request_tokens': Total number of tokens in inputs/prompts  
            - 'response_tokens': Total number of tokens in responses
            - 'thoughts_tokens': Total number of tokens used for internal reasoning
            - 'total_tokens': Total number of tokens used (request + response + thoughts)
            - 'input_cost': Total cost for input tokens
            - 'output_cost': Total cost for output tokens
            - 'thoughts_cost': Total cost for thought tokens
            - 'total_cost': Total cost (input + output + thoughts)
        """
        with self._usage_lock:
            # Create a copy of the usage stats
            usage_copy = self._usage_stats.copy()
        
        # Calculate costs
        input_cost = usage_copy['request_tokens'] * input_ppm_cost / 1_000_000
        output_cost = usage_copy['response_tokens'] * output_ppm_cost / 1_000_000
        thoughts_cost = usage_copy['thoughts_tokens'] * thought_ppm_cost / 1_000_000
        total_cost = input_cost + output_cost + thoughts_cost
        
        # Add cost fields to the copy
        usage_copy['input_cost'] = input_cost
        usage_copy['output_cost'] = output_cost
        usage_copy['thoughts_cost'] = thoughts_cost
        usage_copy['total_cost'] = total_cost
        
        return usage_copy


# Convenience functions for manual cleanup
def cleanup_all_agents() -> None:
    """Manually cleanup all AgentRunner instances and tasks"""
    _task_tracker.cleanup_all()


def cancel_all_running_tasks() -> None:
    """Cancel all currently running asyncio tasks"""
    _task_tracker.cancel_all_tasks()
