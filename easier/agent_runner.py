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


class AgentRunner:
    """
    Concurrent batch processor for EZAgent prompts with optional database persistence.
    
    **AgentRunner** enables efficient processing of multiple prompts using an EZAgent
    with built-in concurrency control, timeout handling, batch processing, and optional
    DuckDB persistence. Designed for high-throughput AI workflows in Jupyter notebooks
    and marimo environments.
    
    ## Key Features
    
    - **Concurrent Processing**: Process multiple prompts simultaneously with configurable concurrency limits
    - **Batch Management**: Automatically splits large prompt lists into manageable batches
    - **Timeout Protection**: Prevents hanging on slow API calls with configurable timeouts
    - **Database Persistence**: Optional DuckDB integration for storing results
    - **Graceful Shutdown**: Handles interrupts and cleanup properly
    - **Context Manager**: Supports `with` statement for automatic resource cleanup
    
    ## Basic Usage
    
    ```python
    import easier as ezr
    
    # Create an EZAgent
    agent = ezr.EZAgent(
        model="gemini-1.5-flash",
        system_prompt="You are a helpful assistant."
    )
    
    # Create and run AgentRunner
    with ezr.AgentRunner(agent) as runner:
        prompts = ["Hello", "What's 2+2?", "Tell me a joke"]
        results = await runner.run(prompts, batch_size=5, max_concurrency=3)
    ```
    
    ## With Database Persistence
    
    ```python
    def my_framer(results):
        import pandas as pd
        data = []
        for i, result in enumerate(results):
            if result is not None:
                data.append({
                    'prompt_id': i,
                    'response': result.data,
                    'model': result.usage().model
                })
        return pd.DataFrame(data)
    
    with ezr.AgentRunner(agent, db_file="results.db") as runner:
        results_df = await runner.run(
            prompts, 
            framer_func=my_framer,
            batch_size=10
        )
    ```
    
    ## Error Handling
    
    The runner gracefully handles:
    - Individual prompt failures (returns None for failed prompts)
    - Timeout errors (configurable per-prompt timeout)
    - Network interruptions (continues with successful results)
    - Keyboard interrupts (Ctrl+C for graceful shutdown)
    
    ## Performance Tips
    
    - Use `batch_size=10-50` for most workloads
    - Set `max_concurrency=5-20` based on API rate limits
    - Enable database persistence for large result sets
    - Use context manager (`with` statement) for proper cleanup
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
        Initialize the AgentRunner with configuration for batch processing and persistence.
        
        ## Parameters
        
        **agent** : `easier.EZAgent`
            The EZAgent instance to use for processing prompts. Must be a properly 
            configured EZAgent with model and system prompt settings.
            
        **db_file** : `str`, optional
            Path to DuckDB database file for result persistence. If provided, results
            will be automatically stored in the database. Requires `framer_func` in 
            the `run()` method to convert results to DataFrame format.
            
        **overwrite** : `bool`, default `False`
            If `True` and `db_file` exists, the existing database will be deleted
            and recreated. Use with caution as this permanently destroys existing data.
            
        **table_name** : `str`, default `"results"`
            Name of the database table to store results. Only relevant when `db_file`
            is provided. Table schema is automatically created based on the first
            DataFrame returned by `framer_func`.
            
        **timeout** : `float`, default `300.0`
            Maximum time in seconds to wait for each individual prompt to complete.
            Prompts exceeding this timeout will be marked as failed (None result)
            and processing will continue with remaining prompts.
        
        ## Examples
        
        **Basic Setup**:
        ```python
        import easier as ezr
        
        agent = ezr.EZAgent(model="gemini-1.5-flash")
        runner = ezr.AgentRunner(agent, timeout=60.0)
        ```
        
        **With Database Persistence**:
        ```python
        runner = ezr.AgentRunner(
            agent=agent,
            db_file="my_results.db",
            table_name="prompt_responses",
            overwrite=True,  # Start fresh
            timeout=120.0
        )
        ```
        
        **Context Manager Usage** (Recommended):
        ```python
        with ezr.AgentRunner(agent, db_file="results.db") as runner:
            # Runner will automatically cleanup on exit
            results = await runner.run(prompts)
        ```
        
        ## Notes
        
        - The runner registers itself with a global task tracker for proper cleanup
        - Database schema is created automatically on first write operation
        - Context manager usage (`with` statement) is recommended for proper resource cleanup
        - If database persistence is enabled, you must provide a `framer_func` in `run()`
        
        ## Raises
        
        **TypeError**
            If `agent` is not an instance of `easier.EZAgent`
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
        
        # Register with global task tracker
        _task_tracker.register_runner(self)

        # Initialize database if enabled
        if self.db_enabled:
            self._init_database()
    
    def __enter__(self) -> "AgentRunner":
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup"""
        self.cleanup()
    
    def cleanup(self) -> None:
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
        self.cleanup()

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

    def create_batches(
        self, prompts: List[str], batch_size: int
    ) -> Generator[List[str], None, None]:
        """Generator that yields batches of prompts"""
        for i in range(0, len(prompts), batch_size):
            yield prompts[i : i + batch_size]

    async def process_batch(
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
                    await self._write_batch_to_db(framed_results, db_write_semaphore)

                return framed_results
            else:
                return results
        except Exception as e:
            print(f"Batch processing failed: {e}")
            raise

    async def _write_batch_to_db(
        self, df: "pd.DataFrame", db_write_semaphore: "asyncio.Semaphore"
    ) -> None:
        """Write batch results to DuckDB"""
        import duckdb

        async with db_write_semaphore:
            try:
                if df.empty:
                    return

                # Create database connection for this batch
                # self.db_file is guaranteed to be not None when db_enabled is True
                assert self.db_file is not None  # Help mypy understand this is not None
                con: "duckdb.DuckDBPyConnection" = duckdb.connect(self.db_file)

                try:
                    con.begin()

                    # Create table if it doesn't exist (using the dataframe schema)
                    con.execute(
                        f"CREATE TABLE IF NOT EXISTS {self.table_name} AS SELECT * FROM df WHERE FALSE"
                    )

                    # Insert the data
                    con.execute(f"INSERT INTO {self.table_name} SELECT * FROM df")

                    con.commit()

                except Exception as db_error:
                    try:
                        con.rollback()
                    except:
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
        Process multiple prompts concurrently with intelligent batch management and error handling.
        
        This is the main method for executing batch AI workflows. It automatically handles
        prompt batching, concurrent execution, timeout management, progress reporting, and
        optional database persistence. Designed for robust, high-throughput processing.
        
        ## Parameters
        
        **prompts** : `List[str]`
            List of prompt strings to process. Each prompt will be sent to the EZAgent
            for processing. Empty or None prompts are processed as-is.
            
        **batch_size** : `int`, default `10`
            Number of prompts to process in each batch. Larger batches reduce overhead
            but increase memory usage. Recommended: 10-50 for most workloads.
            
        **max_concurrency** : `int`, default `10`  
            Maximum number of batches to process simultaneously. Higher values increase
            throughput but may hit API rate limits. Recommended: 5-20 depending on API limits.
            
        **framer_func** : `Callable`, optional
            Function to convert batch results into pandas DataFrame format. Required when
            database persistence is enabled. Should accept `List[Optional[AgentRunResult]]`
            and return `pd.DataFrame`.
            
        **output_type** : `Any`, optional
            Expected output type for structured responses. Passed directly to the
            EZAgent's `run()` method. Used for Pydantic model validation.
        
        ## Returns
        
        **DataFrame or List**
            - If `framer_func` is provided: Returns `pd.DataFrame` with concatenated results
            - If `framer_func` is None: Returns `List[Optional[AgentRunResult]]` with raw results
            - Failed prompts are represented as `None` in the results
        
        ## Examples
        
        **Basic Usage**:
        ```python
        import easier as ezr
        
        agent = ezr.EZAgent(model="gemini-1.5-flash")
        with ezr.AgentRunner(agent) as runner:
            prompts = ["Hello", "What's 2+2?", "Tell me a joke"]
            results = await runner.run(
                prompts, 
                batch_size=5, 
                max_concurrency=3
            )
        ```
        
        **With Custom Framer Function**:
        ```python
        def my_framer(batch_results):
            import pandas as pd
            data = []
            for i, result in enumerate(batch_results):
                if result is not None:
                    data.append({
                        'prompt_num': i,
                        'response': result.data,
                        'tokens_used': result.usage().total_tokens,
                        'model': result.usage().model
                    })
                else:
                    data.append({
                        'prompt_num': i,
                        'response': None,
                        'tokens_used': 0,
                        'model': None
                    })
            return pd.DataFrame(data)
        
        results_df = await runner.run(
            prompts,
            framer_func=my_framer,
            batch_size=20
        )
        ```
        
        **With Structured Output**:
        ```python
        from pydantic import BaseModel
        
        class Analysis(BaseModel):
            sentiment: str
            confidence: float
            
        results = await runner.run(
            sentiment_prompts,
            output_type=Analysis,
            batch_size=15
        )
        ```
        
        **Large Scale Processing**:
        ```python
        # Process 1000s of prompts efficiently
        large_prompts = [f"Analyze text {i}" for i in range(5000)]
        
        with ezr.AgentRunner(agent, db_file="large_results.db") as runner:
            df = await runner.run(
                large_prompts,
                batch_size=50,        # Larger batches for efficiency
                max_concurrency=15,   # Higher concurrency
                framer_func=my_framer
            )
        ```
        
        ## Behavior & Error Handling
        
        **Progress Reporting**: Prints batch completion status in real-time
        
        **Timeout Handling**: Individual prompts timing out are marked as `None` 
        and processing continues with remaining prompts
        
        **Graceful Interruption**: Ctrl+C cancels remaining batches but returns 
        results from completed batches
        
        **API Failures**: Individual prompt failures don't stop the entire run;
        failed prompts return `None` in the results
        
        **Database Persistence**: When `db_file` is configured in `__init__`,
        results are automatically saved to DuckDB as they complete
        
        ## Performance Optimization
        
        **Batch Size Guidelines**:
        - Small prompts (< 100 tokens): `batch_size=30-50`
        - Medium prompts (100-500 tokens): `batch_size=15-25` 
        - Large prompts (> 500 tokens): `batch_size=5-15`
        
        **Concurrency Guidelines**:
        - OpenAI: `max_concurrency=5-10` (rate limit friendly)
        - Gemini: `max_concurrency=10-20` (higher limits)
        - Local models: `max_concurrency=1-5` (hardware limited)
        
        ## Raises
        
        **ValueError**
            If database persistence is enabled (`db_file` provided in `__init__`) 
            but `framer_func` is None
            
        **asyncio.CancelledError**  
            If the operation is cancelled via interrupt or timeout
        """

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
            self.cleanup()

        # Register signal handlers
        original_sigint: Any = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm: Any = signal.signal(signal.SIGTERM, signal_handler)
        
        # Mark as running
        self._is_running = True

        try:
            # Split prompts into batches
            batches: List[List[str]] = list(self.create_batches(prompts, batch_size))
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
                        ] = await self.process_batch(
                            batch, db_write_semaphore, framer_func, output_type
                        )
                        print(f"Finished batch {batch_idx+1} of {total_batches}")
                        return result
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
            self.cleanup()
            
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)


# Convenience functions for manual cleanup
def cleanup_all_agents() -> None:
    """Manually cleanup all AgentRunner instances and tasks"""
    _task_tracker.cleanup_all()


def cancel_all_running_tasks() -> None:
    """Cancel all currently running asyncio tasks"""
    _task_tracker.cancel_all_tasks()
