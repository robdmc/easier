"""Tests for the EZAgent module focusing on integration testing with minimal mocks.

This test suite covers:
- Agent instantiation and configuration
- Cost calculation accuracy
- Error handling and validation
- Core functionality integration

Prefers real objects over mocks wherever possible.
"""

import pytest
import pandas as pd
from pydantic import ValidationError
from typing import Dict, Any

import easier as ezr
from easier.ez_agent import EZAgent, CostConfig, UsageData, MODEL_COSTS


class SimpleUsageResult:
    """Simple result object with usage data for testing cost calculations"""
    def __init__(self, requests=1, request_tokens=10, response_tokens=5, 
                 thoughts_tokens=0, total_tokens=None):
        self.requests = requests
        self.request_tokens = request_tokens
        self.response_tokens = response_tokens
        self.thoughts_tokens = thoughts_tokens
        self.total_tokens = total_tokens or (request_tokens + response_tokens + thoughts_tokens)
    
    def usage(self):
        """Return usage data in expected format"""
        usage = type('Usage', (), {})()
        usage.requests = self.requests
        usage.request_tokens = self.request_tokens
        usage.response_tokens = self.response_tokens
        usage.total_tokens = self.total_tokens
        usage.details = {"reasoning_tokens": self.thoughts_tokens}
        return usage


@pytest.mark.parametrize("model_name", [
    "gpt-4o", "google-vertex:gemini-2.5-flash", "claude-3-5-sonnet-latest"
])
class TestAgentInstantiation:
    """Test EZAgent instantiation across all supported models"""
    
    def test_agent_creation_with_valid_models(self, model_name):
        """Test agent creation works for all supported models"""
        agent = EZAgent("You are helpful", model_name=model_name)
        assert isinstance(agent, EZAgent)
        assert agent.model_name == model_name
    
    def test_agent_creation_with_max_tokens(self, model_name):
        """Test agent creation with max_tokens parameter"""
        agent = EZAgent("You are helpful", model_name=model_name, max_tokens=100)
        assert isinstance(agent, EZAgent)
        assert agent.model_name == model_name


class TestAgentDefaults:
    """Test default behavior and error handling"""
    
    def test_default_model_selection(self):
        """Test default model when none specified"""
        agent = EZAgent("You are helpful")
        assert agent.model_name == "google-vertex:gemini-2.5-flash"
        
        agent_none = EZAgent("You are helpful", model_name=None)
        assert agent_none.model_name == "google-vertex:gemini-2.5-flash"
    
    def test_error_for_unsupported_model(self):
        """Test error handling for unsupported models"""
        with pytest.raises(ValueError, match="Model 'unsupported-model' not supported"):
            EZAgent("You are helpful", model_name="unsupported-model")
    
    def test_error_message_includes_available_models(self):
        """Test error message lists all available models"""
        try:
            EZAgent("You are helpful", model_name="invalid-model")
        except ValueError as e:
            error_msg = str(e)
            # Should include models from all providers
            assert "gpt-4o" in error_msg
            assert "google-vertex:gemini-2.5-flash" in error_msg
            assert "claude-3-5-sonnet-latest" in error_msg
    
    def test_validation_bypass(self):
        """Test validation can be bypassed for custom models"""
        agent = EZAgent("Test", model_name="openai:custom-model", validate_model_name=False)
        assert agent.model_name == "openai:custom-model"


class TestConfiguration:
    """Test cost configuration and validation"""
    
    def test_model_costs_registry(self):
        """Test MODEL_COSTS registry is properly configured"""
        assert len(MODEL_COSTS) > 0
        
        # Verify all entries are valid CostConfig instances
        for model_name, config in MODEL_COSTS.items():
            assert isinstance(config, CostConfig)
            assert config.input_ppm_cost >= 0
            assert config.output_ppm_cost >= 0
            assert config.thought_ppm_cost >= 0
    
    def test_cost_config_validation(self):
        """Test CostConfig Pydantic validation"""
        # Valid config
        config = CostConfig(input_ppm_cost=1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0)
        assert config.input_ppm_cost == 1.0
        
        # Negative costs should fail
        with pytest.raises(ValidationError):
            CostConfig(input_ppm_cost=-1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0)
    
    def test_usage_data_validation(self):
        """Test UsageData Pydantic validation"""
        # Valid data
        data = UsageData(requests=1, request_tokens=10, response_tokens=5, 
                        thoughts_tokens=2, total_tokens=17)
        assert data.requests == 1
        assert data.total_tokens == 17
        
        # Negative values should fail
        with pytest.raises(ValidationError):
            UsageData(requests=-1, request_tokens=10, response_tokens=5, 
                     thoughts_tokens=2, total_tokens=17)


class TestUsageAndCostCalculation:
    """Test usage tracking and cost calculation with simple data"""
    
    def test_usage_calculation_single_result(self):
        """Test usage calculation with single result"""
        agent = EZAgent("Test", model_name="gpt-4o")
        result = SimpleUsageResult(requests=1, request_tokens=8, response_tokens=3, 
                                 thoughts_tokens=1, total_tokens=12)
        
        usage = agent.get_usage(result)
        
        assert isinstance(usage, pd.Series)
        assert usage['requests'] == 1
        assert usage['request_tokens'] == 8
        assert usage['response_tokens'] == 3
        assert usage['thoughts_tokens'] == 1
        assert usage['total_tokens'] == 12
    
    def test_usage_calculation_multiple_results(self):
        """Test usage aggregation across multiple results"""
        agent = EZAgent("Test", model_name="gpt-4o")
        result1 = SimpleUsageResult(requests=1, request_tokens=8, response_tokens=3, thoughts_tokens=1)
        result2 = SimpleUsageResult(requests=1, request_tokens=15, response_tokens=6, thoughts_tokens=3)
        
        usage = agent.get_usage(result1, result2)
        
        assert usage['requests'] == 2
        assert usage['request_tokens'] == 23  # 8 + 15
        assert usage['response_tokens'] == 9   # 3 + 6
        assert usage['thoughts_tokens'] == 4   # 1 + 3
    
    def test_get_usage_no_results_error(self):
        """Test error when no results provided"""
        agent = EZAgent("Test", model_name="gpt-4o")
        with pytest.raises(ValueError, match="At least one result must be provided"):
            agent.get_usage()

    @pytest.mark.parametrize("model_name,request_tokens,response_tokens,thoughts_tokens", [
        ("gpt-4o", 8, 3, 1),
        ("google-vertex:gemini-2.5-flash", 12, 4, 0),
        ("claude-3-5-sonnet-latest", 10, 2, 0),
    ])
    def test_cost_calculation_accuracy(self, model_name, request_tokens, response_tokens, thoughts_tokens):
        """Test cost calculations are mathematically accurate across models"""
        agent = EZAgent("Test", model_name=model_name)
        result = SimpleUsageResult(requests=1, request_tokens=request_tokens,
                                 response_tokens=response_tokens, thoughts_tokens=thoughts_tokens)
        
        costs = agent.get_cost(result)
        
        assert isinstance(costs, pd.Series)
        assert 'input_cost' in costs
        assert 'output_cost' in costs
        assert 'thoughts_cost' in costs
        assert 'total_cost' in costs
        
        # Verify mathematical accuracy
        model_config = MODEL_COSTS[model_name]
        expected_input = request_tokens * model_config.input_ppm_cost / 1_000_000
        expected_output = response_tokens * model_config.output_ppm_cost / 1_000_000
        expected_thoughts = thoughts_tokens * model_config.thought_ppm_cost / 1_000_000
        expected_total = expected_input + expected_output + expected_thoughts
        
        assert abs(costs['input_cost'] - expected_input) < 1e-10
        assert abs(costs['output_cost'] - expected_output) < 1e-10
        assert abs(costs['thoughts_cost'] - expected_thoughts) < 1e-10
        assert abs(costs['total_cost'] - expected_total) < 1e-10
    
    def test_cost_calculation_multiple_results(self):
        """Test cost aggregation across multiple results"""
        agent = EZAgent("Test", model_name="gpt-4o-mini")
        result1 = SimpleUsageResult(requests=1, request_tokens=8, response_tokens=3, thoughts_tokens=1)
        result2 = SimpleUsageResult(requests=1, request_tokens=20, response_tokens=10, thoughts_tokens=5)
        
        costs = agent.get_cost(result1, result2)
        
        # Should aggregate: 8+20=28 request, 3+10=13 response, 1+5=6 thoughts
        model_config = MODEL_COSTS["gpt-4o-mini"]
        expected_total = ((28 * model_config.input_ppm_cost) + 
                         (13 * model_config.output_ppm_cost) + 
                         (6 * model_config.thought_ppm_cost)) / 1_000_000
        
        assert abs(costs['total_cost'] - expected_total) < 1e-10
    
    def test_get_cost_no_results_error(self):
        """Test error when no results provided"""
        agent = EZAgent("Test", model_name="gpt-4o")
        with pytest.raises(ValueError, match="At least one result must be provided"):
            agent.get_cost()


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_token_usage(self):
        """Test cost calculation with zero tokens"""
        agent = EZAgent("Test", model_name="gpt-4o")
        result = SimpleUsageResult(requests=1, request_tokens=0, response_tokens=0, 
                                 thoughts_tokens=0, total_tokens=0)
        
        costs = agent.get_cost(result)
        
        assert costs['input_cost'] == 0.0
        assert costs['output_cost'] == 0.0
        assert costs['thoughts_cost'] == 0.0
        assert costs['total_cost'] == 0.0
    
    def test_cost_missing_model_error(self):
        """Test error when model cost config is not available"""
        agent = EZAgent("Test", model_name="gpt-4o", validate_model_name=False)
        agent.cost_config = None  # Simulate missing cost config
        agent.model_name = "nonexistent-model"  # Set to invalid model name for error message
        
        result = SimpleUsageResult()
        with pytest.raises(KeyError, match="Model 'nonexistent-model' not found"):
            agent.get_cost(result)
    
    def test_agent_missing_model_name_attribute(self):
        """Test error when agent has no model_name attribute"""
        agent = EZAgent("Test", model_name="gpt-4o")
        delattr(agent, 'model_name')
        
        result = SimpleUsageResult()
        with pytest.raises(KeyError, match="Model 'unknown' not found"):
            agent.get_cost(result)


class TestIntegrationWorkflow:
    """Test complete workflows"""
    
    def test_agent_creation_to_cost_calculation(self):
        """Test complete workflow from creation to cost calculation"""
        agent = EZAgent("You are helpful", model_name="gpt-4o-mini")
        assert isinstance(agent, EZAgent)
        
        result = SimpleUsageResult(requests=1, request_tokens=100, response_tokens=50, 
                                 thoughts_tokens=10, total_tokens=160)
        
        # Test usage tracking
        usage = agent.get_usage(result)
        assert usage['requests'] == 1
        assert usage['total_tokens'] == 160
        
        # Test cost calculation
        costs = agent.get_cost(result)
        assert costs['total_cost'] > 0
        assert costs['input_cost'] > 0
        assert costs['output_cost'] > 0


class TestBackwardCompatibility:
    """Test that core behavior remains unchanged"""
    
    def test_usage_return_format(self):
        """Test that get_usage returns expected pandas Series format"""
        agent = EZAgent("Test", model_name="gpt-4o")
        result = SimpleUsageResult()
        
        usage = agent.get_usage(result)
        
        assert isinstance(usage, pd.Series)
        expected_keys = {'requests', 'request_tokens', 'response_tokens', 'thoughts_tokens', 'total_tokens'}
        assert set(usage.index) == expected_keys
        assert all(pd.api.types.is_numeric_dtype(type(usage[key])) for key in expected_keys)
    
    def test_cost_return_format(self):
        """Test that get_cost returns expected pandas Series format"""
        agent = EZAgent("Test", model_name="gpt-4o")
        result = SimpleUsageResult()
        
        costs = agent.get_cost(result)
        
        assert isinstance(costs, pd.Series)
        expected_keys = {'input_cost', 'output_cost', 'thoughts_cost', 'total_cost'}
        assert set(costs.index) == expected_keys