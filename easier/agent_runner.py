"""
AgentRunner module for processing prompts with EZAgent in batches with concurrency control.

This module provides the AgentRunner class that can process multiple prompts
concurrently using an EZAgent, with optional database persistence using DuckDB.
"""

import asyncio
import atexit
import inspect
import os
import signal
import threading
import time
import weakref
from typing import Callable, Any, Generator, Union, List, Optional, TYPE_CHECKING, Set, Dict

if TYPE_CHECKING:
    import easier  # type: ignore
    import pandas as pd
    import pydantic_ai.agent
    import duckdb

from pydantic import BaseModel




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
        if hasattr(self, "_initialized") and self._initialized:
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
            # Only print if there's actually something to clean up
            if self.active_tasks or self.active_runners:
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


class DuckDBTypeConverter:
    """Handles Python to DuckDB type mapping for database schema generation"""

    # Type mapping dictionaries for DuckDB schema generation
    BASIC_TYPE_MAPPINGS = {
        type(None): "VARCHAR",
        int: "INTEGER",
        float: "DOUBLE",
        bool: "BOOLEAN",
        str: "VARCHAR",
        bytes: "BLOB",
    }

    GENERIC_ORIGIN_MAPPINGS = {list: "JSON", dict: "JSON", tuple: "JSON"}

    def handle_union_type(self, python_type) -> Optional[str]:
        """Handle Union and Optional types"""
        import typing

        if not hasattr(python_type, "__origin__") or python_type.__origin__ is not typing.Union:
            return None

        args = python_type.__args__
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], get the non-None type
            non_none_type = next(arg for arg in args if arg is not type(None))
            return self.map_python_type_to_db(non_none_type)
        else:
            # Complex union - store as JSON to preserve flexibility
            return "JSON"

    def handle_special_types(self, python_type) -> Optional[str]:
        """Handle Enum, Pydantic models, and datetime types"""
        from enum import Enum

        # Handle Enum types
        if hasattr(python_type, "__bases__") and any(
            issubclass(base, Enum) for base in python_type.__bases__ if isinstance(base, type)
        ):
            return "VARCHAR"

        # Handle Pydantic models (have model_fields attribute)
        if hasattr(python_type, "model_fields"):
            return "JSON"

        # Handle datetime types
        try:
            import datetime

            if python_type in (datetime.datetime, datetime.date, datetime.time):
                return "TIMESTAMP"
        except ImportError:
            pass

        return None

    def map_python_type_to_db(self, python_type) -> str:
        """Map Python/Pydantic types to DuckDB column types"""

        # 1. Check basic types first (fastest lookup)
        if python_type in self.BASIC_TYPE_MAPPINGS:
            return self.BASIC_TYPE_MAPPINGS[python_type]

        # 2. Handle generic types with __origin__
        if hasattr(python_type, "__origin__"):
            origin = python_type.__origin__

            # Check Union types (including Optional)
            union_result = self.handle_union_type(python_type)
            if union_result is not None:
                return union_result

            # Check generic origin mappings
            if origin in self.GENERIC_ORIGIN_MAPPINGS:
                return self.GENERIC_ORIGIN_MAPPINGS[origin]

        # 3. Handle special types (Enum, Pydantic, datetime)
        special_result = self.handle_special_types(python_type)
        if special_result is not None:
            return special_result

        # 4. Default: complex or unknown types become JSON
        return "JSON"


class TableSchemaStrategy:
    """Handles different table schema creation scenarios for database persistence"""

    def __init__(self, map_python_type_to_duckdb_func: Callable[[Any], str]):
        """Initialize with a function to map Python types to DuckDB types"""
        self.map_python_type_to_duckdb = map_python_type_to_duckdb_func

    def create_schema_from_pydantic(self, output_type: Any) -> List[str]:
        """Create schema parts from a Pydantic model"""
        schema_parts = []
        model_fields = output_type.model_fields if hasattr(output_type, "model_fields") else {}

        for field_name, field_info in model_fields.items():
            field_type = field_info.annotation if hasattr(field_info, "annotation") else str
            db_type = self.map_python_type_to_duckdb(field_type)
            schema_parts.append(f'"{field_name}" {db_type}')

        return schema_parts

    def should_create_table_from_pydantic(self, table_exists: bool, output_type: Any, df_empty: bool) -> bool:
        """Determine if table should be created using Pydantic schema"""
        return not table_exists and output_type is not None

    def should_create_table_from_dataframe(self, table_exists: bool, output_type: Any, df_empty: bool) -> bool:
        """Determine if table should be created from DataFrame structure"""
        return not table_exists and output_type is None and not df_empty

    def should_skip_table_creation(self, table_exists: bool, output_type: Any, df_empty: bool) -> bool:
        """Determine if table creation should be skipped"""
        return not table_exists and output_type is None and df_empty


class DataPreprocessor:
    """Handles DataFrame preprocessing for database insertion"""

    def filter_columns_by_pydantic_model(self, df: "pd.DataFrame", output_type: Any) -> "pd.DataFrame":
        """Filter DataFrame to only include columns from Pydantic model"""
        if output_type is None or not hasattr(output_type, "model_fields"):
            return df.copy()

        model_columns = list(output_type.model_fields.keys())
        available_columns = [col for col in model_columns if col in df.columns]
        
        # Warn about columns that will be dropped
        dropped_columns = [col for col in df.columns if col not in model_columns]
        if dropped_columns:
            import warnings
            warnings.warn(
                f"DataFrame contains columns {dropped_columns} that are not in the Pydantic model "
                f"'{output_type.__name__}'. These columns will be dropped during database persistence. "
                f"Consider adding them to your Pydantic model or setting output_type=None to preserve all columns.",
                UserWarning,
                stacklevel=3
            )
        
        # Use loc to ensure we always get a DataFrame
        return df.loc[:, available_columns].copy()

    def serialize_json_fields(self, df: "pd.DataFrame", output_type: Any) -> "pd.DataFrame":
        """Serialize fields that should be JSON (lists, dicts) to JSON strings"""
        if output_type is None or not hasattr(output_type, "model_fields"):
            return df

        import json
        import pandas as pd

        df_processed = df.copy()

        for field_name, field_info in output_type.model_fields.items():
            if field_name in df_processed.columns:
                field_type = field_info.annotation if hasattr(field_info, "annotation") else str

                if hasattr(field_type, "__origin__"):
                    try:
                        origin = getattr(field_type, "__origin__", None)
                        if origin in (list, dict):
                            series = pd.Series(df_processed[field_name])
                            df_processed[field_name] = series.apply(lambda x: json.dumps(x) if x is not None else None)
                    except AttributeError:
                        pass

        return df_processed

    def prepare_dataframe_for_insertion(self, df: "pd.DataFrame", output_type: Any) -> "pd.DataFrame":
        """Complete preprocessing pipeline for DataFrame insertion"""
        if df.empty:
            return df

        if output_type is not None and hasattr(output_type, "model_fields"):
            df_filtered = self.filter_columns_by_pydantic_model(df, output_type)
            df_processed = self.serialize_json_fields(df_filtered, output_type)
            return df_processed
        else:
            return df.copy()


class DatabaseErrorHandler:
    """Handles database error formatting and debugging information collection"""

    def get_table_schema(self, connection: "duckdb.DuckDBPyConnection", table_name: str) -> Union[Dict[str, str], str]:
        """Get table schema for debugging purposes"""
        try:
            schema_result = connection.execute(f"DESCRIBE {table_name}").fetchall()
            return {row[0]: row[1] for row in schema_result}
        except Exception:
            return "Unable to retrieve table schema"

    def get_dataframe_dtypes(self, df: "pd.DataFrame") -> Dict[str, str]:
        """Get DataFrame dtypes for debugging purposes"""
        return {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}

    def get_dataframe_sample(self, df: "pd.DataFrame") -> Union[Dict[str, Any], str]:
        """Get DataFrame sample for debugging purposes"""
        if len(df) > 0:
            return df.iloc[0].to_dict()
        else:
            return "Empty DataFrame"

    def create_schema_validation_error(
        self, original_error: Exception, table_name: str, connection: "duckdb.DuckDBPyConnection", df: "pd.DataFrame"
    ) -> DatabaseSchemaValidationError:
        """Create a detailed DatabaseSchemaValidationError with debugging info"""
        table_schema = self.get_table_schema(connection, table_name)
        df_dtypes = self.get_dataframe_dtypes(df)
        df_sample = self.get_dataframe_sample(df)

        error_msg = (
            f"Database type conversion failed for table '{table_name}':\n"
            f"Original error: {original_error}\n"
            f"Table schema: {table_schema}\n"
            f"DataFrame dtypes: {df_dtypes}\n"
            f"DataFrame sample (first row): {df_sample}"
        )

        return DatabaseSchemaValidationError(error_msg)


class ValidationHandler:
    """Handles input validation for the run method"""

    def __init__(self, runner: "AgentRunner"):
        self.runner = runner

    def validate_context_manager(self) -> None:
        """Validate that the runner is being used within a context manager"""
        if not self.runner._in_context:
            raise RuntimeError(
                "AgentRunner.run() must be called within a context manager. "
                "Use 'with AgentRunner(agent) as runner:' before calling run()."
            )

    def validate_database_config(self, framer_func: Optional[Callable]) -> None:
        """Validate database configuration requirements"""
        if self.runner.db_enabled and framer_func is None:
            raise ValueError(
                f"Database file '{self.runner.db_file}' is configured but framer_func is not provided. "
                "Both db_file and framer_func are required for database persistence."
            )


class SignalManager:
    """Handles signal management and graceful shutdown coordination"""

    def __init__(self, runner: "AgentRunner", logger_func: Callable[[str], None]):
        self.runner = runner
        self.log_info = logger_func
        self.shutdown_event: Optional[asyncio.Event] = None
        self.original_sigint: Any = None
        self.original_sigterm: Any = None

    def setup_shutdown_handling(self) -> asyncio.Event:
        """Set up signal handlers and return shutdown event"""
        self.shutdown_event = asyncio.Event()

        def signal_handler(signum: int, frame: Any) -> None:
            self.log_info(f"Received signal {signum}, initiating immediate shutdown...")
            if self.shutdown_event:
                self.shutdown_event.set()
            # Cancel all tasks immediately
            _task_tracker.cancel_all_tasks()
            self.runner._cleanup()

        # Register signal handlers
        self.original_sigint = signal.signal(signal.SIGINT, signal_handler)
        self.original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

        return self.shutdown_event

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers"""
        if self.original_sigint is not None:
            signal.signal(signal.SIGINT, self.original_sigint)
        if self.original_sigterm is not None:
            signal.signal(signal.SIGTERM, self.original_sigterm)


class ProgressTracker:
    """Handles progress tracking, timing, cost estimation, and logging"""

    def __init__(
        self, runner: "AgentRunner", logger_func: Callable[[str], None], format_duration_func: Callable[[float], str]
    ):
        self.runner = runner
        self.log_info = logger_func
        self.format_duration = format_duration_func
        self.completion_counter: int = 0
        self.completion_lock: asyncio.Lock = asyncio.Lock()
        self.start_time: Optional[float] = None

    def initialize_tracking(
        self, total_prompts: int, total_batches: int, batch_size: int, max_concurrency: int
    ) -> None:
        """Initialize progress tracking"""
        self.runner._is_running = True
        self.start_time = time.time()
        self.runner._start_time = self.start_time

        self.log_info(
            f"Processing {total_prompts} prompts in {total_batches} batches "
            f"(batch_size={batch_size}, max_concurrency={max_concurrency})"
        )

    async def track_batch_completion(self, total_batches: int) -> None:
        """Track completion of a batch and log progress"""
        async with self.completion_lock:
            self.completion_counter += 1
            current_completed_batches = self.completion_counter

        # Get current usage stats with costs in an async-safe way
        usage_stats = self.runner.get_usage()
        current_cost = float(usage_stats.get("total_cost", 0.0) or 0.0)

        # Calculate timing and cost estimations based on actual completions
        current_time = time.time()
        elapsed_time = current_time - (self.start_time or current_time)
        completed_batches = current_completed_batches

        # Estimate total time and cost based on completed batches
        estimated_total_time = (
            elapsed_time * total_batches / completed_batches if completed_batches > 0 else elapsed_time
        )
        estimated_total_cost = (
            current_cost * total_batches / completed_batches
            if completed_batches > 0 and current_cost > 0
            else current_cost
        )

        # Format durations
        elapsed_str = self.format_duration(elapsed_time)
        estimated_str = self.format_duration(estimated_total_time)

        # Static padding widths for consistent alignment
        TIME_WIDTH = 12  # Allow for estimates like "9999h 59m 59s"
        BATCH_WIDTH = 6  # Allow for up to 999,999 batches
        COST_WIDTH = 6

        # Format with static padding for consistent alignment (left-justified)
        padded_batch = str(completed_batches).ljust(BATCH_WIDTH)
        padded_elapsed = elapsed_str.ljust(TIME_WIDTH)
        padded_estimated = estimated_str.ljust(TIME_WIDTH)
        padded_current_cost = f"{current_cost:.2f}".ljust(COST_WIDTH)
        padded_estimated_cost = f"{estimated_total_cost:.2f}".ljust(COST_WIDTH)

        self.log_info(
            f"completed batch: {padded_batch} of {str(total_batches).ljust(BATCH_WIDTH)}, "
            f"time: {padded_elapsed} of {padded_estimated}, "
            f"cost: ${padded_current_cost} of ${padded_estimated_cost}"
        )

    def log_final_summary(self, successful_batches: int, failed_batches: int, total_batches: int) -> None:
        """Log final execution summary"""
        self.log_info(
            f"Processed {successful_batches}/{total_batches} batches successfully. {failed_batches} batches failed."
        )

        # Log final cost summary with breakdown by token type
        final_usage = self.runner.get_usage()
        self.log_info(
            f"Cost Summary - Input: ${final_usage.get('input_cost', 0.0):.4f}, "
            f"Output: ${final_usage.get('output_cost', 0.0):.4f}, "
            f"Thoughts: ${final_usage.get('thoughts_cost', 0.0):.4f}, "
            f"Total: ${final_usage.get('total_cost', 0.0):.4f}"
        )

        # Log total running time summary
        if self.start_time is not None:
            total_elapsed_time = time.time() - self.start_time
            total_elapsed_str = self.format_duration(total_elapsed_time)
            self.log_info(f"Total Running Time: {total_elapsed_str}")
        else:
            self.log_info("Total Running Time: Unable to calculate (start time not recorded)")


class TaskManager:
    """Handles async task creation, tracking, and cleanup"""

    def __init__(self, runner: "AgentRunner", logger_func: Callable[[str], None]):
        self.runner = runner
        self.log_info = logger_func

    def create_batch_tasks(
        self, batches: List[List[str]], batch_processor_func: Callable[[List[str], int], Any]
    ) -> List[asyncio.Task]:
        """Create async tasks for all batches with proper tracking"""
        batch_tasks: List[asyncio.Task] = []

        for idx, batch in enumerate(batches):
            task = asyncio.create_task(batch_processor_func(batch, idx))
            batch_tasks.append(task)

            # Track tasks in both local and global trackers
            self.runner.active_tasks.add(task)
            _task_tracker.add_task(task)

            # Remove from local tracker when done
            def cleanup_local_task(t: asyncio.Task, runner_ref: "AgentRunner" = self.runner) -> None:
                runner_ref.active_tasks.discard(t)

            task.add_done_callback(cleanup_local_task)

        return batch_tasks

    async def handle_keyboard_interrupt(
        self, batch_tasks: List[asyncio.Task], total_batches: int
    ) -> List[Optional["pydantic_ai.agent.AgentRunResult"]]:
        """Handle keyboard interrupt by cancelling tasks gracefully"""
        self.log_info("Received interrupt signal, cancelling all tasks...")

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
            self.log_info("Some tasks did not cancel gracefully within timeout")

        # Count completed batches
        completed_results: List[Any] = [r for r in batch_tasks if r.done() and not r.cancelled()]
        self.log_info(f"Processing interrupted. Completed {len(completed_results)}/{total_batches} batches.")

        # For interrupted execution, return flattened list
        interrupted_results: List[Optional["pydantic_ai.agent.AgentRunResult"]] = []
        for r in completed_results:
            if not isinstance(r.result(), Exception) and r.result() is not None:
                if isinstance(r.result(), list):
                    interrupted_results.extend(r.result())
                else:
                    # If it's a DataFrame, we can't flatten it properly here
                    # This is a limitation of the interrupted execution path
                    pass
        return interrupted_results


class ResultAggregator:
    """Handles result processing and DataFrame concatenation"""

    def count_batch_results(self, results: List[Any]) -> tuple[int, int]:
        """Count successful and failed batches"""
        successful_batches = sum(1 for result in results if not isinstance(result, Exception) and result is not None)
        failed_batches = sum(1 for result in results if isinstance(result, Exception) or result is None)
        return successful_batches, failed_batches

    def aggregate_results(
        self, results: List[Any], framer_func: Optional[Callable]
    ) -> Union["pd.DataFrame", List[Optional["pydantic_ai.agent.AgentRunResult"]]]:
        """Aggregate results into final format based on framer_func"""
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
            flattened_results: List[Optional["pydantic_ai.agent.AgentRunResult"]] = []
            for result in results:
                if result is not None and not isinstance(result, Exception):
                    flattened_results.extend(result)
            return flattened_results


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
        logger=None,
        type_converter: Optional[DuckDBTypeConverter] = None,
    ) -> None:
        """
        Initialize AgentRunner for batch processing with optional database persistence.

        Args:
            agent: EZAgent instance for processing prompts
            db_file: Path to DuckDB database file for result persistence (optional)
            overwrite: If True, delete existing database file (default False)
            table_name: Database table name for results (default "results")
            timeout: Max seconds per prompt before timeout (default 300.0)
            logger: Logger instance for progress messages (default None disables logging)

        Example:
            with ezr.AgentRunner(agent, db_file="results.db", timeout=60.0) as runner:
                results = await runner.run(prompts)

        Note: Database persistence requires framer_func in run() method.
        """
        import easier as ezr

        if not isinstance(agent, ezr.EZAgent):
            raise TypeError(f"Expected agent to be of type ezr.EZAgent, got {type(agent)}")
        self.agent: "easier.EZAgent" = agent

        # Database configuration
        self.db_enabled: bool = db_file is not None
        self.db_file: Optional[str] = db_file
        self.overwrite: bool = overwrite
        self.table_name: str = table_name

        # Timeout configuration
        self.timeout: float = timeout


        # Logger configuration
        self.logger = logger

        # Task management
        self.active_tasks: Set[asyncio.Task] = set()
        self._is_running: bool = False
        self._cleanup_done: bool = False
        self._in_context: bool = False

        # Usage tracking
        import pandas as pd

        self._usage_stats: "pd.Series" = pd.Series(
            {"requests": 0, "request_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0}
        )
        self._usage_lock: threading.Lock = threading.Lock()

        # Timing tracking
        self._start_time: Optional[float] = None

        # Register with global task tracker
        _task_tracker.register_runner(self)

        # Initialize type converter (injectable for database flexibility)
        self._type_converter = type_converter or DuckDBTypeConverter()

        # Initialize database helper components
        self._schema_strategy = TableSchemaStrategy(self._type_converter.map_python_type_to_db)
        self._data_preprocessor = DataPreprocessor()
        self._error_handler = DatabaseErrorHandler()

        # Initialize run method helper components
        self._validation_handler = ValidationHandler(self)
        self._signal_manager = SignalManager(self, self._log_info)
        self._progress_tracker = ProgressTracker(self, self._log_info, self._format_duration)
        self._task_manager = TaskManager(self, self._log_info)
        self._result_aggregator = ResultAggregator()

        # Initialize database if enabled
        if self.db_enabled:
            self._init_database()

    def _log_info(self, message: str) -> None:
        """Log a message if logger is configured, otherwise do nothing"""
        if self.logger is not None:
            self.logger.info(message)

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to a readable string (e.g., '1h 23m 45s', '2m 34s', '45s')"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining_seconds = int(seconds % 60)
            return f"{hours}h {minutes}m {remaining_seconds}s"

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

        self._log_info(f"AgentRunner cleanup: Cancelling {len(self.active_tasks)} tasks...")

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
                self._log_info(f"Deleted existing {self.db_file} for overwrite.")
            except FileNotFoundError:
                pass

        # Connect and create table structure based on first framed result
        # Since we don't know the schema yet, we'll defer table creation to first write

    def _ensure_table_exists(
        self, connection: "duckdb.DuckDBPyConnection", df: "pd.DataFrame", output_type: Any
    ) -> None:
        """Ensure the database table exists, creating it if necessary"""
        # Check if table exists
        table_exists_result = connection.execute(
            f"""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = '{self.table_name}'
        """
        ).fetchone()
        table_exists = table_exists_result[0] > 0 if table_exists_result else False

        if table_exists:
            return

        df_empty = df.empty

        if self._schema_strategy.should_create_table_from_pydantic(table_exists, output_type, df_empty):
            schema_parts = self._schema_strategy.create_schema_from_pydantic(output_type)
            if schema_parts:
                create_sql = f"CREATE TABLE {self.table_name} ({', '.join(schema_parts)})"
                connection.execute(create_sql)
            else:
                connection.execute(f"CREATE TABLE {self.table_name} AS SELECT * FROM df WHERE FALSE")

        elif self._schema_strategy.should_create_table_from_dataframe(table_exists, output_type, df_empty):
            connection.execute(f"CREATE TABLE {self.table_name} AS SELECT * FROM df WHERE FALSE")

        elif self._schema_strategy.should_skip_table_creation(table_exists, output_type, df_empty):
            pass

    def _prepare_dataframe_for_insert(self, df: "pd.DataFrame", output_type: Any) -> "pd.DataFrame":
        """Prepare DataFrame for database insertion"""
        return self._data_preprocessor.prepare_dataframe_for_insertion(df, output_type)

    def _execute_database_insert(self, connection: "duckdb.DuckDBPyConnection", df_to_insert: "pd.DataFrame") -> None:
        """Execute the database insertion"""
        if not df_to_insert.empty:
            connection.execute(f"INSERT INTO {self.table_name} SELECT * FROM df_to_insert")

    def _validate_run_parameters(self, framer_func: Optional[Callable]) -> None:
        """Validate run method parameters"""
        self._validation_handler.validate_context_manager()
        self._validation_handler.validate_database_config(framer_func)

    def _setup_execution_context(
        self, prompts: List[str], batch_size: int, max_concurrency: int
    ) -> tuple[List[List[str]], int, asyncio.Semaphore, Optional[asyncio.Semaphore]]:
        """Setup execution context and return necessary components"""
        # Split prompts into batches
        batches = list(self._create_batches(prompts, batch_size))
        total_batches = len(batches)

        # Create semaphores for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        db_write_semaphore = asyncio.Semaphore(1) if self.db_enabled else None

        # Initialize progress tracking
        self._progress_tracker.initialize_tracking(len(prompts), total_batches, batch_size, max_concurrency)

        return batches, total_batches, semaphore, db_write_semaphore

    def _create_batch_processor(
        self,
        semaphore: asyncio.Semaphore,
        db_write_semaphore: Optional[asyncio.Semaphore],
        framer_func: Optional[Callable],
        output_type: Any,
        total_batches: int,
        callback_func: Optional[Callable],
        callback_ctx: Any,
    ) -> Callable[[List[str], int], Any]:
        """Create the batch processing function with proper closure"""

        async def process_batch_with_semaphore(
            batch: List[str], batch_idx: int
        ) -> Optional[Union["pd.DataFrame", List[Optional["pydantic_ai.agent.AgentRunResult"]]]]:
            try:
                async with semaphore:
                    result = await self._process_batch(
                        batch, db_write_semaphore, framer_func, output_type, callback_func, callback_ctx
                    )

                    # Track batch completion and log progress
                    await self._progress_tracker.track_batch_completion(total_batches)

                    return result
            except DatabaseSchemaValidationError as e:
                # Re-raise schema validation errors as they should fail fast
                self._log_info(f"Failed to process batch {batch_idx+1}: {e}")
                raise
            except Exception as e:
                self._log_info(f"Failed to process batch {batch_idx+1}: {e}")
                return None

        return process_batch_with_semaphore

    async def _execute_batch_processing(self, batch_tasks: List[asyncio.Task]) -> List[Any]:
        """Execute batch processing and handle exceptions"""
        try:
            # Wait for all batches to complete
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Check for DatabaseSchemaValidationError and re-raise it
            for result in results:
                if isinstance(result, DatabaseSchemaValidationError):
                    raise result

            return results
        except KeyboardInterrupt:
            # Handle keyboard interrupt with task manager
            return await self._task_manager.handle_keyboard_interrupt(batch_tasks, len(batch_tasks))

    def _handle_execution_results(
        self, results: List[Any], framer_func: Optional[Callable]
    ) -> Union["pd.DataFrame", List[Optional["pydantic_ai.agent.AgentRunResult"]]]:
        """Handle and aggregate execution results"""
        successful_batches, failed_batches = self._result_aggregator.count_batch_results(results)
        total_batches = successful_batches + failed_batches

        # Log final summary
        self._progress_tracker.log_final_summary(successful_batches, failed_batches, total_batches)

        # Aggregate and return results
        return self._result_aggregator.aggregate_results(results, framer_func)

    def _create_batches(self, prompts: List[str], batch_size: int) -> Generator[List[str], None, None]:
        """Generator that yields batches of prompts"""
        for i in range(0, len(prompts), batch_size):
            yield prompts[i : i + batch_size]

    async def _process_batch(
        self,
        batch_prompts: List[str],
        db_write_semaphore: Optional["asyncio.Semaphore"] = None,
        framer_func: Optional[Callable[[List[Optional["pydantic_ai.agent.AgentRunResult"]]], "pd.DataFrame"]] = None,
        output_type: Any = None,
        callback_func: Optional[Callable] = None,
        callback_ctx: Any = None,
    ) -> Union["pd.DataFrame", List[Optional["pydantic_ai.agent.AgentRunResult"]]]:
        """Process a batch of prompts with timeout support"""
        try:
            results: List[Optional["pydantic_ai.agent.AgentRunResult"]] = []
            for prompt in batch_prompts:
                try:
                    # Add timeout wrapper for each LLM call
                    result: Optional["pydantic_ai.agent.AgentRunResult"] = await asyncio.wait_for(
                        self.agent.run(prompt, output_type=output_type), timeout=self.timeout
                    )
                    results.append(result)

                    # Execute callback on successful result
                    if result is not None:
                        self._execute_callback(callback_func, callback_ctx, result)

                    # Aggregate usage stats in a thread-safe manner
                    try:
                        usage_series = self.agent.get_usage(result)
                        with self._usage_lock:
                            self._usage_stats += usage_series
                    except Exception as usage_error:
                        self._log_info(f"Failed to aggregate usage stats: {usage_error}")
                except asyncio.TimeoutError:
                    self._log_info(f"Prompt timed out after {self.timeout}s: {prompt[:100]}...")
                    results.append(None)
                except Exception as prompt_error:
                    self._log_info(f"Prompt failed: {prompt_error}")
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
            self._log_info(f"Batch processing failed: {e}")
            raise

    async def _write_batch_to_db(
        self, df: "pd.DataFrame", db_write_semaphore: "asyncio.Semaphore", output_type: Any = None
    ) -> None:
        """Write batch results to DuckDB using dependency-injected components"""
        import duckdb

        async with db_write_semaphore:
            try:
                # Create database connection for this batch
                assert self.db_file is not None  # Help mypy understand this is not None
                con: "duckdb.DuckDBPyConnection" = duckdb.connect(self.db_file)

                try:
                    # Ensure table exists using strategy pattern
                    self._ensure_table_exists(con, df, output_type)

                    # Skip processing if DataFrame is empty and table was skipped
                    if df.empty and self._schema_strategy.should_skip_table_creation(False, output_type, True):
                        return

                    # Prepare DataFrame for insertion using data preprocessor
                    df_to_insert = self._prepare_dataframe_for_insert(df, output_type)

                    # Execute the database insertion
                    try:
                        self._execute_database_insert(con, df_to_insert)
                    except Exception as conversion_error:
                        # Use error handler to create detailed error message
                        detailed_error = self._error_handler.create_schema_validation_error(
                            conversion_error, self.table_name, con, df_to_insert
                        )
                        raise detailed_error from conversion_error

                except Exception as db_error:
                    try:
                        con.rollback()
                    except Exception:
                        pass
                    raise db_error
                finally:
                    con.close()

            except Exception as e:
                self._log_info(f"Database write failed: {e}")
                raise

    async def run(
        self,
        prompts: List[str],
        batch_size: int = 10,
        max_concurrency: int = 10,
        framer_func: Optional[Callable[[List[Optional["pydantic_ai.agent.AgentRunResult"]]], "pd.DataFrame"]] = None,
        output_type: Any = None,
        callback_func: Optional[Callable] = None,
        callback_ctx: Any = None,
    ) -> Union["pd.DataFrame", List[Optional["pydantic_ai.agent.AgentRunResult"]]]:
        """
        Process multiple prompts concurrently with batch management and error handling.

        Args:
            prompts: List of prompt strings to process
            batch_size: Number of prompts per batch (default 10, recommend 10-50)
            max_concurrency: Max simultaneous batches (default 10, recommend 5-20)
            framer_func: Function to convert results to DataFrame (required for DB persistence)
            output_type: Expected output type for structured responses (optional)
            callback_func: Optional callback function to execute on each successful result.
                          Signature can be either func(result) or func(result, ctx)
            callback_ctx: Optional context object passed to callback_func if it accepts ctx parameter

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

            # With callback function
            def save_result(result, ctx=None):
                # Save each result to JSON file
                with open(f"result_{ctx['counter']}.json", "w") as f:
                    json.dump(result.model_dump(), f)
                ctx['counter'] += 1

            with ezr.AgentRunner(agent) as runner:
                results = await runner.run(
                    prompts,
                    callback_func=save_result,
                    callback_ctx={'counter': 0}
                )

        Note: Handles timeouts, interrupts, and API failures gracefully.
        Prints progress and saves to database if configured.
        Callback failures are logged but don't stop processing.
        """

        # Validate input parameters
        self._validate_run_parameters(framer_func)

        # Set up signal handling and execution context
        self._signal_manager.setup_shutdown_handling()
        batches, total_batches, semaphore, db_write_semaphore = self._setup_execution_context(
            prompts, batch_size, max_concurrency
        )

        try:
            # Create batch processor function
            batch_processor = self._create_batch_processor(
                semaphore, db_write_semaphore, framer_func, output_type, total_batches, callback_func, callback_ctx
            )

            # Create and track async tasks for all batches
            batch_tasks = self._task_manager.create_batch_tasks(batches, batch_processor)

            # Execute batch processing with exception handling
            results = await self._execute_batch_processing(batch_tasks)

            # Handle and aggregate results
            return self._handle_execution_results(results, framer_func)

        finally:
            # Ensure cleanup and restore signal handlers
            self._is_running = False
            self._cleanup()
            self._signal_manager.restore_signal_handlers()

    def get_usage(self) -> "pd.Series":
        """
        Get aggregated usage statistics with cost calculations.

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

        # Calculate costs using the MODEL_COSTS registry
        if hasattr(self.agent, 'model_name'):
            try:
                from easier.ez_agent import MODEL_COSTS
                if self.agent.model_name in MODEL_COSTS:
                    model_config = MODEL_COSTS[self.agent.model_name]
                    input_cost = usage_copy["request_tokens"] * model_config.input_ppm_cost / 1_000_000
                    output_cost = usage_copy["response_tokens"] * model_config.output_ppm_cost / 1_000_000
                    thoughts_cost = usage_copy["thoughts_tokens"] * model_config.thought_ppm_cost / 1_000_000
                else:
                    # Fallback to zero costs if model not found in MODEL_COSTS
                    input_cost = output_cost = thoughts_cost = 0.0
            except ImportError:
                # Fallback for cases where MODEL_COSTS is not available
                input_cost = output_cost = thoughts_cost = 0.0
        else:
            # Fallback for agents without model_name
            input_cost = output_cost = thoughts_cost = 0.0
        
        total_cost = input_cost + output_cost + thoughts_cost

        # Add cost fields to the copy
        usage_copy["input_cost"] = input_cost
        usage_copy["output_cost"] = output_cost
        usage_copy["thoughts_cost"] = thoughts_cost
        usage_copy["total_cost"] = total_cost

        return usage_copy

    def _execute_callback(
        self, callback_func: Callable, callback_ctx: Any, result: "pydantic_ai.agent.AgentRunResult"
    ) -> None:
        """
        Execute callback function on a successful result with proper signature detection.

        Args:
            callback_func: The callback function to execute
            callback_ctx: Context object to pass if callback accepts it
            result: The successful AgentRunResult to pass to callback

        Note: Handles callback exceptions gracefully with logging.
        """
        if callback_func is None:
            return

        try:
            # Inspect the callback function signature
            sig = inspect.signature(callback_func)
            params = list(sig.parameters.keys())

            # Check if callback accepts a 'ctx' parameter
            if len(params) >= 2 and ('ctx' in params or 'callback_ctx' in params):
                # Call with context
                callback_func(result, ctx=callback_ctx)
            elif len(params) >= 2:
                # Call with context using positional argument
                callback_func(result, callback_ctx)
            else:
                # Call without context
                callback_func(result)

        except Exception as callback_error:
            self._log_info(f"Callback execution failed: {callback_error}")


# Convenience functions for manual cleanup
def cleanup_all_agents() -> None:
    """Manually cleanup all AgentRunner instances and tasks"""
    _task_tracker.cleanup_all()


def cancel_all_running_tasks() -> None:
    """Cancel all currently running asyncio tasks"""
    _task_tracker.cancel_all_tasks()
