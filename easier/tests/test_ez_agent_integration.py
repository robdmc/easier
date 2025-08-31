"""
Integration tests for EZAgent using real API calls.

These tests make actual API calls to OpenAI and Gemini models to verify:
- Factory pattern works with real agents
- Cost calculations are accurate with real usage data
- Usage tracking works correctly
- Both agent types integrate properly

Note: These tests require valid API keys and will incur small costs.
"""

import pytest
import pandas as pd
import asyncio
from typing import List

import easier as ezr
from easier.ez_agent import EZAgent, MODEL_COSTS


# Fixture to provide all models from MODEL_COSTS registry
def get_all_models():
    """Get all models from MODEL_COSTS registry"""
    return list(MODEL_COSTS.keys())

@pytest.fixture(params=get_all_models())
def model_name(request):
    """Fixture that yields every model from all agent types"""
    return request.param


@pytest.mark.integration
class TestHelloWorldAllAgentTypes:
    """Hello world test for one representative model from each agent type"""
    
    @pytest.mark.asyncio
    async def test_hello_world_all_agent_types(self, model_name):
        """Test basic 'hello world' functionality to verify API and configuration works"""
        agent = EZAgent("You are a helpful assistant.", model_name=model_name)
        
        result = await agent.run("Say hello")
        
        # Verify we got a response - this ensures API call succeeded
        assert result is not None
        assert hasattr(result, 'output')
        assert result.output is not None
        assert len(str(result.output)) > 0, f"Empty response from model {model_name}"


@pytest.mark.integration
class TestFactoryPatternIntegration:
    """Test EZAgent factory pattern with real API calls"""

    def test_factory_creates_openai_agent(self):
        """Test factory creates OpenAI agent and it works"""
        agent = EZAgent("You are a helpful assistant. Answer briefly.", model_name="gpt-4o-mini")
        assert isinstance(agent, EZAgent)
        assert agent.model_name == "gpt-4o-mini"

    def test_factory_creates_gemini_agent(self):
        """Test factory creates Gemini agent and it works"""
        agent = EZAgent("You are a helpful assistant. Answer briefly.", model_name="google-vertex:gemini-2.5-flash")
        assert isinstance(agent, EZAgent)
        assert agent.model_name == "google-vertex:gemini-2.5-flash"

    def test_factory_creates_anthropic_agent(self):
        """Test factory creates Anthropic agent and it works"""
        agent = EZAgent("You are a helpful assistant. Answer briefly.", model_name="claude-3-5-haiku-latest")
        assert isinstance(agent, EZAgent)
        assert agent.model_name == "claude-3-5-haiku-latest"

    @pytest.mark.asyncio
    async def test_openai_agent_real_call(self):
        """Test OpenAI agent with real API call"""
        agent = EZAgent("You are a helpful math assistant. Give very brief answers.", model_name="gpt-4o-mini")
        
        result = await agent.run("What is 2+2?")
        
        # Verify we got a response
        assert result is not None
        assert hasattr(result, 'output')
        assert result.output is not None
        assert "4" in str(result.output)

    @pytest.mark.asyncio
    async def test_gemini_agent_real_call(self):
        """Test Gemini agent with real API call"""
        agent = EZAgent("You are a helpful assistant. Give very brief answers.", model_name="google-vertex:gemini-2.5-flash")
        
        result = await agent.run("What color is the sky?")
        
        # Verify we got a response
        assert result is not None
        assert hasattr(result, 'output')
        assert result.output is not None
        assert len(str(result.output)) > 0

    @pytest.mark.asyncio
    async def test_anthropic_agent_real_call(self):
        """Test Anthropic agent with real API call"""
        agent = EZAgent("You are a helpful assistant. Give very brief answers.", model_name="claude-3-5-haiku-latest")
        
        result = await agent.run("What is the capital of France?")
        
        # Verify we got a response
        assert result is not None
        assert hasattr(result, 'output')
        assert result.output is not None
        assert "Paris" in str(result.output)


@pytest.mark.integration
class TestUsageTrackingIntegration:
    """Test usage tracking with real API responses"""

    @pytest.mark.asyncio
    async def test_openai_usage_tracking_real(self):
        """Test OpenAI usage tracking with real API call"""
        agent = EZAgent("Answer in exactly 5 words.", model_name="gpt-4o-mini")
        
        result = await agent.run("What is 10 + 15?")
        
        # Test usage tracking
        usage = agent.get_usage(result)
        
        assert isinstance(usage, pd.Series)
        assert usage['requests'] == 1
        assert usage['request_tokens'] > 0  # Should have consumed input tokens
        assert usage['response_tokens'] > 0  # Should have generated output tokens
        assert usage['total_tokens'] > 0
        assert usage['total_tokens'] == usage['request_tokens'] + usage['response_tokens'] + usage['thoughts_tokens']

    @pytest.mark.asyncio
    async def test_gemini_usage_tracking_real(self):
        """Test Gemini usage tracking with real API call"""
        agent = EZAgent("Answer in exactly 3 words.", model_name="google-vertex:gemini-2.5-flash")
        
        result = await agent.run("What is 5 + 7?")
        
        # Test usage tracking
        usage = agent.get_usage(result)
        
        assert isinstance(usage, pd.Series)
        assert usage['requests'] == 1
        assert usage['request_tokens'] > 0  # Should have consumed input tokens
        assert usage['response_tokens'] > 0  # Should have generated output tokens
        assert usage['total_tokens'] > 0
        assert usage['total_tokens'] == usage['request_tokens'] + usage['response_tokens'] + usage['thoughts_tokens']

    @pytest.mark.asyncio
    async def test_anthropic_usage_tracking_real(self):
        """Test Anthropic usage tracking with real API call"""
        agent = EZAgent("Answer in exactly 3 words.", model_name="claude-3-5-haiku-latest")
        
        result = await agent.run("What is 8 + 9?")
        
        # Test usage tracking
        usage = agent.get_usage(result)
        
        assert isinstance(usage, pd.Series)
        assert usage['requests'] == 1
        assert usage['request_tokens'] > 0  # Should have consumed input tokens
        assert usage['response_tokens'] > 0  # Should have generated output tokens
        assert usage['total_tokens'] > 0
        assert usage['total_tokens'] == usage['request_tokens'] + usage['response_tokens'] + usage['thoughts_tokens']

    @pytest.mark.asyncio
    async def test_usage_aggregation_multiple_calls(self):
        """Test usage aggregation across multiple real API calls"""
        agent = EZAgent("Answer very briefly.", model_name="gpt-4o-mini")
        
        # Make multiple calls
        result1 = await agent.run("1+1=?")
        result2 = await agent.run("2+2=?")
        result3 = await agent.run("3+3=?")
        
        # Test aggregated usage
        usage = agent.get_usage(result1, result2, result3)
        
        assert usage['requests'] == 3
        assert usage['request_tokens'] > 0
        assert usage['response_tokens'] > 0
        assert usage['total_tokens'] > 0
        
        # Should be more tokens than single call
        single_usage = agent.get_usage(result1)
        assert usage['total_tokens'] > single_usage['total_tokens']


@pytest.mark.integration
class TestCostCalculationIntegration:
    """Test cost calculations with real API usage data"""

    @pytest.mark.asyncio
    async def test_openai_cost_calculation_real(self):
        """Test OpenAI cost calculation with real usage data"""
        agent = EZAgent("Answer very briefly in 2-3 words.", model_name="gpt-4o-mini")
        
        result = await agent.run("What is Python?")
        
        # Test cost calculation
        costs = agent.get_cost(result)
        
        assert isinstance(costs, pd.Series)
        assert 'input_cost' in costs
        assert 'output_cost' in costs
        assert 'thoughts_cost' in costs
        assert 'total_cost' in costs
        
        # Should have real costs (gpt-4o-mini: input=0.15, output=0.6 per million)
        assert costs['input_cost'] > 0  # Should have input cost
        assert costs['output_cost'] > 0  # Should have output cost
        assert costs['total_cost'] > 0
        assert costs['total_cost'] == costs['input_cost'] + costs['output_cost'] + costs['thoughts_cost']

    @pytest.mark.asyncio
    async def test_gemini_cost_calculation_real(self):
        """Test Gemini cost calculation with real usage data"""
        agent = EZAgent("Answer very briefly.", model_name="google-vertex:gemini-2.5-flash")
        
        result = await agent.run("What is JavaScript?")
        
        # Test cost calculation
        costs = agent.get_cost(result)
        
        assert isinstance(costs, pd.Series)
        
        # Should have real costs (gemini-2.5-flash: input=0.5, output=3.0 per million)
        assert costs['input_cost'] > 0  # Should have input cost
        assert costs['output_cost'] > 0  # Should have output cost
        assert costs['total_cost'] > 0
        assert costs['total_cost'] == costs['input_cost'] + costs['output_cost'] + costs['thoughts_cost']

    @pytest.mark.asyncio
    async def test_anthropic_cost_calculation_real(self):
        """Test Anthropic cost calculation with real usage data"""
        agent = EZAgent("Answer very briefly.", model_name="claude-3-5-haiku-latest")
        
        result = await agent.run("What is TypeScript?")
        
        # Test cost calculation
        costs = agent.get_cost(result)
        
        assert isinstance(costs, pd.Series)
        
        # Should have real costs (claude-3-5-haiku-latest: input=1.0, output=5.0 per million)
        assert costs['input_cost'] > 0  # Should have input cost
        assert costs['output_cost'] > 0  # Should have output cost
        assert costs['total_cost'] > 0
        assert costs['total_cost'] == costs['input_cost'] + costs['output_cost'] + costs['thoughts_cost']

    @pytest.mark.asyncio
    async def test_cost_accuracy_openai(self):
        """Test cost calculation accuracy for OpenAI"""
        agent = EZAgent("Answer in exactly 1 word.", model_name="gpt-4o-mini")
        
        result = await agent.run("Yes or no?")
        
        usage = agent.get_usage(result)
        costs = agent.get_cost(result)
        
        # Manual calculation to verify accuracy
        model_config = MODEL_COSTS["gpt-4o-mini"]
        expected_input = usage['request_tokens'] * model_config.input_ppm_cost / 1_000_000
        expected_output = usage['response_tokens'] * model_config.output_ppm_cost / 1_000_000
        expected_thoughts = usage['thoughts_tokens'] * model_config.thought_ppm_cost / 1_000_000
        expected_total = expected_input + expected_output + expected_thoughts
        
        # Verify calculations match
        assert abs(costs['input_cost'] - expected_input) < 1e-10
        assert abs(costs['output_cost'] - expected_output) < 1e-10
        assert abs(costs['thoughts_cost'] - expected_thoughts) < 1e-10
        assert abs(costs['total_cost'] - expected_total) < 1e-10

    @pytest.mark.asyncio
    async def test_cost_accuracy_gemini(self):
        """Test cost calculation accuracy for Gemini"""
        agent = EZAgent("Answer in exactly 1 word.", model_name="google-vertex:gemini-2.5-flash")
        
        result = await agent.run("True or false?")
        
        usage = agent.get_usage(result)
        costs = agent.get_cost(result)
        
        # Manual calculation to verify accuracy
        model_config = MODEL_COSTS["google-vertex:gemini-2.5-flash"]
        expected_input = usage['request_tokens'] * model_config.input_ppm_cost / 1_000_000
        expected_output = usage['response_tokens'] * model_config.output_ppm_cost / 1_000_000
        expected_thoughts = usage['thoughts_tokens'] * model_config.thought_ppm_cost / 1_000_000
        expected_total = expected_input + expected_output + expected_thoughts
        
        # Verify calculations match
        assert abs(costs['input_cost'] - expected_input) < 1e-10
        assert abs(costs['output_cost'] - expected_output) < 1e-10
        assert abs(costs['thoughts_cost'] - expected_thoughts) < 1e-10
        assert abs(costs['total_cost'] - expected_total) < 1e-10

    @pytest.mark.asyncio
    async def test_cost_accuracy_anthropic(self):
        """Test cost calculation accuracy for Anthropic"""
        agent = EZAgent("Answer in exactly 1 word.", model_name="claude-3-5-haiku-latest")
        
        result = await agent.run("Good or bad?")
        
        usage = agent.get_usage(result)
        costs = agent.get_cost(result)
        
        # Manual calculation to verify accuracy
        model_config = MODEL_COSTS["claude-3-5-haiku-latest"]
        expected_input = usage['request_tokens'] * model_config.input_ppm_cost / 1_000_000
        expected_output = usage['response_tokens'] * model_config.output_ppm_cost / 1_000_000
        expected_thoughts = usage['thoughts_tokens'] * model_config.thought_ppm_cost / 1_000_000
        expected_total = expected_input + expected_output + expected_thoughts
        
        # Verify calculations match
        assert abs(costs['input_cost'] - expected_input) < 1e-10
        assert abs(costs['output_cost'] - expected_output) < 1e-10
        assert abs(costs['thoughts_cost'] - expected_thoughts) < 1e-10
        assert abs(costs['total_cost'] - expected_total) < 1e-10


@pytest.mark.integration
class TestAgentRunnerIntegration:
    """Test AgentRunner integration with real agents and API calls"""

    @pytest.mark.asyncio
    async def test_agent_runner_openai_integration(self):
        """Test AgentRunner with real OpenAI agent"""
        from easier.agent_runner import AgentRunner
        
        agent = EZAgent("Answer very briefly in 1-2 words.", model_name="gpt-4o-mini")
        
        with AgentRunner(agent) as runner:
            results = await runner.run(["What is 1+1?", "What is 2+2?"])
            
            # Should get results
            assert len(results) == 2
            assert all(result is not None for result in results)
            
            # Should have usage stats
            usage = runner.get_usage()
            assert usage['requests'] > 0
            assert usage['request_tokens'] > 0
            assert usage['response_tokens'] > 0
            assert usage['total_cost'] > 0

    @pytest.mark.asyncio
    async def test_agent_runner_gemini_integration(self):
        """Test AgentRunner with real Gemini agent"""
        from easier.agent_runner import AgentRunner
        
        agent = EZAgent("Answer very briefly in 1-2 words.", model_name="google-vertex:gemini-2.5-flash")
        
        with AgentRunner(agent) as runner:
            results = await runner.run(["What is 3+3?", "What is 4+4?"])
            
            # Should get results
            assert len(results) == 2
            assert all(result is not None for result in results)
            
            # Should have usage stats
            usage = runner.get_usage()
            assert usage['requests'] > 0
            assert usage['request_tokens'] > 0
            assert usage['response_tokens'] > 0
            assert usage['total_cost'] > 0

    @pytest.mark.asyncio
    async def test_cost_consistency_agent_vs_runner(self):
        """Test that agent.get_cost() and runner.get_usage() give consistent results"""
        from easier.agent_runner import AgentRunner
        
        agent = EZAgent("Answer briefly.", model_name="gpt-4o-mini")
        
        with AgentRunner(agent) as runner:
            # Run single prompt
            results = await runner.run(["What is 5+5?"])
            result = results[0]
            
            # Get costs from agent directly
            agent_costs = agent.get_cost(result)
            
            # Get costs from runner
            runner_usage = runner.get_usage()
            
            # Should be very close (allowing for tiny floating point differences)
            assert abs(agent_costs['input_cost'] - runner_usage['input_cost']) < 1e-8
            assert abs(agent_costs['output_cost'] - runner_usage['output_cost']) < 1e-8
            assert abs(agent_costs['thoughts_cost'] - runner_usage['thoughts_cost']) < 1e-8
            assert abs(agent_costs['total_cost'] - runner_usage['total_cost']) < 1e-8


@pytest.mark.integration
class TestCrossModelComparison:
    """Compare behavior between OpenAI and Gemini models"""

    @pytest.mark.asyncio
    async def test_same_prompt_different_models(self):
        """Test same prompt with both OpenAI and Gemini"""
        openai_agent = EZAgent("Answer very briefly.", model_name="gpt-4o-mini")
        gemini_agent = EZAgent("Answer very briefly.", model_name="google-vertex:gemini-2.5-flash")
        
        prompt = "What is machine learning?"
        
        openai_result = await openai_agent.run(prompt)
        gemini_result = await gemini_agent.run(prompt)
        
        # Both should work
        assert openai_result is not None
        assert gemini_result is not None
        
        # Test usage tracking
        openai_usage = openai_agent.get_usage(openai_result)
        gemini_usage = gemini_agent.get_usage(gemini_result)
        
        # Both should have positive usage
        assert openai_usage['total_tokens'] > 0
        assert gemini_usage['total_tokens'] > 0
        
        # Test cost calculations
        openai_costs = openai_agent.get_cost(openai_result)
        gemini_costs = gemini_agent.get_cost(gemini_result)
        
        # Both should have costs
        assert openai_costs['total_cost'] > 0
        assert gemini_costs['total_cost'] > 0
        
        # Cost structures should be the same
        assert set(openai_costs.index) == set(gemini_costs.index)

    @pytest.mark.asyncio
    async def test_cost_per_token_differences(self):
        """Verify that different models have different cost structures"""
        openai_agent = EZAgent("Answer briefly.", model_name="gpt-4o-mini")
        gemini_agent = EZAgent("Answer briefly.", model_name="google-vertex:gemini-2.5-flash")
        
        # Get model configs
        openai_config = MODEL_COSTS["gpt-4o-mini"]
        gemini_config = MODEL_COSTS["google-vertex:gemini-2.5-flash"]
        
        # Should have different cost structures
        # gpt-4o-mini: input=0.15, output=0.6
        # gemini-2.5-flash: input=0.5, output=3.0
        assert openai_config.input_ppm_cost != gemini_config.input_ppm_cost
        assert openai_config.output_ppm_cost != gemini_config.output_ppm_cost
        
        # Run same prompt to compare actual costs
        prompt = "List 3 colors."
        
        openai_result = await openai_agent.run(prompt)
        gemini_result = await gemini_agent.run(prompt)
        
        openai_costs = openai_agent.get_cost(openai_result)
        gemini_costs = gemini_agent.get_cost(gemini_result)
        
        # Should have different costs (models have different pricing)
        # Note: Actual costs will depend on token usage, so we just verify they're both positive
        assert openai_costs['total_cost'] > 0
        assert gemini_costs['total_cost'] > 0