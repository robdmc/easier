#! /usr/bin/env python3

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full", sql_output="polars")

with app.setup:
    # Initialization code that runs before all other cells# Setup cell code
    import marimo as mo
    import pandas as pd
    import holoviews as hv
    import json
    import os
    import easier as ezr
    from typing import List

    # from duckcacher import DuckCacher, duckloader_factory
    from pydantic import BaseModel, Field

    # import logfire

    # Requires you have a .logfire directory populated
    # logfire.configure()
    # logfire.instrument_pydantic_ai()

    import logging

    # Set the specific logger to a higher level to silence debug messages
    logging.getLogger("google_genai.models").setLevel(logging.CRITICAL)
    logger = ezr.get_logger("llm")


@app.cell
def _():
    SYSTEM_PROMPT = """
    You are a friendly chap from Scotland. You will be given one or more numbers.  For each number state some facts in a jovial scotish manner
    """
    return (SYSTEM_PROMPT,)


@app.cell
def _():
    ld = ezr.duckloader_factory("stuff_from_replica.ddb")
    dfc = ld.df_cases
    dfc

    return (dfc,)


@app.cell
def _():
    os.getcwd()
    return


@app.cell
def _(dfc):
    def get_prompts(df):
        prompts = []
        for tup in df.itertuples():
            prompts.append(f"Tell me three things you know about about the number {tup.id}.")

        return prompts

    def framer_func(results):
        return pd.DataFrame(r.output.model_dump() for r in results)

    prompts = get_prompts(dfc)
    prompts
    # prompts = [p for p in prompts if '1_' in p]
    # prompts
    # prompts = prompts[:10]
    return framer_func, prompts


@app.class_definition
class Answer(BaseModel):
    number: int = Field(description="The number to consider")
    facts: List[str] = Field(description="things I know about this number")


@app.cell
async def _(SYSTEM_PROMPT, prompts):
    async def run_single(prompt):
        agent = ezr.EZAgent(model_name="google-vertex:gemini-2.5-flash", system_prompt=SYSTEM_PROMPT)
        return await agent.run(prompt, output_type=Answer)

    res = await run_single(prompts[0])
    res.output.model_dump()
    return


@app.cell
async def _(SYSTEM_PROMPT, framer_func, prompts):
    async def run_multi(prompts):
        agent = ezr.EZAgent(model_name="google-vertex:gemini-2.5-flash", system_prompt=SYSTEM_PROMPT)
        with ezr.AgentRunner(agent=agent, db_file="final_result.ddb", overwrite=True, logger=logger) as runner:
            results = await runner.run(
                prompts=prompts,
                batch_size=1,
                max_concurrency=10,
                framer_func=framer_func,
                output_type=Answer,
            )

    await run_multi(prompts)
    return


@app.cell
def _():
    ldr = ezr.duckloader_factory("final_result.ddb")
    ldr.ls()
    return (ldr,)


@app.cell
def _(ldr):
    dfr = ldr.df_results
    dfr["facts"] = dfr.facts.map(json.loads)
    dfr = dfr.explode("facts").reset_index(drop=True)
    dfr
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
