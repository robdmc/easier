"""
Tests for the EZAgent module using mock agents to avoid external API calls.

This test suite covers the EZAgent class hierarchy including:
- Factory pattern functionality
- Model configuration and validation
- Usage tracking and cost calculation
- Error handling scenarios
- Pydantic validation
"""

import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Optional, Any
from pydantic import BaseModel, ValidationError

import easier as ezr
from easier.ez_agent import EZAgent, OpenAIAgent, GeminiAgent, ModelConfig, UsageData


class MockResult:
    """Mock agent run result for testing"""
    def __init__(self, output="test response", usage_data=None):
        self.output = output
        self._usage_data = usage_data or {
            "requests": 1,
            "request_tokens": 10, 
            "response_tokens": 5,
            "thoughts_tokens": 2,
            "total_tokens": 17
        }
        
    def usage(self):
        """Mock usage method that returns usage data"""
        mock_usage = Mock()
        mock_usage.requests = self._usage_data["requests"]
        mock_usage.request_tokens = self._usage_data["request_tokens"] 
        mock_usage.response_tokens = self._usage_data["response_tokens"]
        mock_usage.total_tokens = self._usage_data["total_tokens"]
        
        # For OpenAI agents, add reasoning_tokens in details
        mock_usage.details = {"reasoning_tokens": self._usage_data["thoughts_tokens"]}
        
        return mock_usage


@pytest.fixture
def mock_openai_result():
    """Mock OpenAI agent result"""
    return MockResult(output="4", usage_data={
        "requests": 1,
        "request_tokens": 8,
        "response_tokens": 3, 
        "thoughts_tokens": 1,
        "total_tokens": 12
    })


@pytest.fixture 
def mock_gemini_result():
    """Mock Gemini agent result"""
    return MockResult(output="Blue", usage_data={
        "requests": 1,
        "request_tokens": 12,
        "response_tokens": 4,
        "thoughts_tokens": 0,  # Gemini may not have thoughts
        "total_tokens": 16
    })


class TestFactoryPattern:
    """Test EZAgent factory pattern functionality"""
    
    def test_factory_selects_openai_agent(self):
        """Test factory creates OpenAIAgent for OpenAI model names"""
        agent = EZAgent("You are helpful", model_name="gpt-4o")
        assert isinstance(agent, OpenAIAgent)
        assert agent.model_name == "gpt-4o"
    
    def test_factory_selects_gemini_agent(self):
        """Test factory creates GeminiAgent for Gemini model names"""
        agent = EZAgent("You are helpful", model_name="google-vertex:gemini-2.5-flash")
        assert isinstance(agent, GeminiAgent)
        assert agent.model_name == "google-vertex:gemini-2.5-flash"
    
    def test_direct_instantiation_still_works(self):
        """Test direct instantiation bypasses factory"""
        openai_agent = OpenAIAgent("You are helpful", model_name="gpt-4o")
        assert isinstance(openai_agent, OpenAIAgent)
        assert openai_agent.model_name == "gpt-4o"
        
        gemini_agent = GeminiAgent("You are helpful", model_name="google-vertex:gemini-2.5-flash")
        assert isinstance(gemini_agent, GeminiAgent)
        assert gemini_agent.model_name == "google-vertex:gemini-2.5-flash"
    
    def test_factory_error_for_unsupported_model(self):
        """Test factory raises ValueError for unsupported models"""
        with pytest.raises(ValueError, match="Model 'unsupported-model' not supported"):
            EZAgent("You are helpful", model_name="unsupported-model")
    
    def test_factory_error_lists_available_models(self):
        """Test factory error message includes available models"""
        try:
            EZAgent("You are helpful", model_name="unsupported-model")
        except ValueError as e:
            error_msg = str(e)
            # Should list both OpenAI and Gemini models
            assert "gpt-4o" in error_msg
            assert "google-vertex:gemini-2.5-flash" in error_msg
    
    def test_factory_with_none_model_name(self):
        """Test factory raises error when model_name is None"""
        with pytest.raises(ValueError, match="Model 'None' not supported"):
            EZAgent("You are helpful", model_name=None)
    
    def test_factory_with_empty_string_model_name(self):
        """Test factory raises error when model_name is empty string"""
        with pytest.raises(ValueError, match="Model '' not supported"):
            EZAgent("You are helpful", model_name="")


class TestModelConfiguration:
    """Test model configuration and allowed_models structure"""
    
    def test_base_class_empty_allowed_models(self):
        """Test base EZAgent class has empty allowed_models"""
        assert EZAgent.allowed_models == {}
    
    def test_openai_agent_allowed_models_structure(self):
        """Test OpenAIAgent has properly structured allowed_models"""
        models = OpenAIAgent.allowed_models
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check some expected models
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-3.5-turbo" in models
        
        # Verify all values are ModelConfig instances
        for model_name, config in models.items():
            assert isinstance(config, ModelConfig)
            assert config.input_ppm_cost >= 0
            assert config.output_ppm_cost >= 0
            assert config.thought_ppm_cost >= 0
    
    def test_gemini_agent_allowed_models_structure(self):
        """Test GeminiAgent has properly structured allowed_models"""
        models = GeminiAgent.allowed_models
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check expected models
        assert "google-vertex:gemini-2.5-flash" in models
        assert "google-vertex:gemini-2.5-pro" in models
        
        # Verify all values are ModelConfig instances
        for model_name, config in models.items():
            assert isinstance(config, ModelConfig)
            assert config.input_ppm_cost >= 0
            assert config.output_ppm_cost >= 0
            assert config.thought_ppm_cost >= 0
    
    def test_model_config_validation(self):
        """Test ModelConfig Pydantic validation"""
        # Valid config should work
        config = ModelConfig(input_ppm_cost=1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0)
        assert config.input_ppm_cost == 1.0
        assert config.output_ppm_cost == 5.0
        assert config.thought_ppm_cost == 5.0
        
        # Negative costs should fail
        with pytest.raises(ValidationError):
            ModelConfig(input_ppm_cost=-1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0)
        
        with pytest.raises(ValidationError):
            ModelConfig(input_ppm_cost=1.0, output_ppm_cost=-5.0, thought_ppm_cost=5.0)
        
        with pytest.raises(ValidationError):
            ModelConfig(input_ppm_cost=1.0, output_ppm_cost=5.0, thought_ppm_cost=-5.0)


class TestValidation:
    """Test model validation functionality"""
    
    def test_openai_model_validation_enabled(self):
        """Test OpenAI model validation when enabled"""
        # Valid model should work
        agent = OpenAIAgent("Test", model_name="gpt-4o", validate_model_name=True)
        assert agent.model_name == "gpt-4o"
        
        # Invalid model should fail
        with pytest.raises(ValueError, match="Model invalid-model not supported"):
            OpenAIAgent("Test", model_name="invalid-model", validate_model_name=True)
    
    def test_gemini_model_validation_enabled(self):
        """Test Gemini model validation when enabled"""
        # Valid model should work
        agent = GeminiAgent("Test", model_name="google-vertex:gemini-2.5-flash", validate_model_name=True)
        assert agent.model_name == "google-vertex:gemini-2.5-flash"
        
        # Invalid model should fail
        with pytest.raises(ValueError, match="Model invalid-model not supported"):
            GeminiAgent("Test", model_name="invalid-model", validate_model_name=True)
    
    def test_model_validation_disabled(self):
        """Test validation can be bypassed"""
        # Should work even with invalid model name (but needs valid pydantic-ai format)
        # Use openai:custom-model which is a valid format but not in our allowed_models
        agent = OpenAIAgent("Test", model_name="openai:custom-model", validate_model_name=False)
        assert agent.model_name == "openai:custom-model"
        
        agent = GeminiAgent("Test", model_name="google-vertex:custom-model", validate_model_name=False)
        assert agent.model_name == "google-vertex:custom-model"


class TestUsageDataValidation:
    """Test UsageData Pydantic validation"""
    
    def test_valid_usage_data(self):
        """Test valid usage data passes validation"""
        data = {
            "requests": 1,
            "request_tokens": 10,
            "response_tokens": 5,
            "thoughts_tokens": 2,
            "total_tokens": 17
        }
        validated = UsageData(**data)
        assert validated.requests == 1
        assert validated.request_tokens == 10
        assert validated.response_tokens == 5
        assert validated.thoughts_tokens == 2
        assert validated.total_tokens == 17
    
    def test_usage_data_negative_values_fail(self):
        """Test negative values fail validation"""
        with pytest.raises(ValidationError):
            UsageData(requests=-1, request_tokens=10, response_tokens=5, thoughts_tokens=2, total_tokens=17)
        
        with pytest.raises(ValidationError):
            UsageData(requests=1, request_tokens=-10, response_tokens=5, thoughts_tokens=2, total_tokens=17)
    
    def test_usage_data_validation_in_agent(self):
        """Test _validate_usage method uses UsageData validation"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        
        # Valid data should work
        valid_data = {
            "requests": 1,
            "request_tokens": 10,
            "response_tokens": 5,
            "thoughts_tokens": 2,
            "total_tokens": 17
        }
        result = agent._validate_usage(valid_data)
        assert result == valid_data
        
        # Invalid data should raise ValidationError
        invalid_data = {
            "requests": -1,  # Negative value
            "request_tokens": 10,
            "response_tokens": 5,
            "thoughts_tokens": 2,
            "total_tokens": 17
        }
        with pytest.raises(ValidationError):
            agent._validate_usage(invalid_data)


class TestUsageTracking:
    """Test usage tracking functionality (backward compatibility)"""
    
    def test_openai_get_usage_single_result(self, mock_openai_result):
        """Test OpenAI get_usage with single result"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        
        usage = agent.get_usage(mock_openai_result)
        
        assert isinstance(usage, pd.Series)
        assert usage['requests'] == 1
        assert usage['request_tokens'] == 8
        assert usage['response_tokens'] == 3
        assert usage['thoughts_tokens'] == 1  # From details.reasoning_tokens
        assert usage['total_tokens'] == 12
    
    def test_gemini_get_usage_single_result(self, mock_gemini_result):
        """Test Gemini get_usage with single result"""
        agent = GeminiAgent("Test", model_name="google-vertex:gemini-2.5-flash")
        
        usage = agent.get_usage(mock_gemini_result)
        
        assert isinstance(usage, pd.Series)
        assert usage['requests'] == 1
        assert usage['request_tokens'] == 12
        assert usage['response_tokens'] == 4
        assert usage['thoughts_tokens'] == 0  # Gemini may not have thoughts
        assert usage['total_tokens'] == 16
    
    def test_get_usage_no_results_raises_error(self):
        """Test get_usage raises ValueError when no results provided"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        
        with pytest.raises(ValueError, match="At least one result must be provided"):
            agent.get_usage()
    
    def test_get_usage_multiple_results(self, mock_openai_result):
        """Test get_usage aggregates multiple results"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        
        # Create a second result with different usage
        result2 = MockResult(output="8", usage_data={
            "requests": 1,
            "request_tokens": 15,
            "response_tokens": 6,
            "thoughts_tokens": 3,
            "total_tokens": 24
        })
        
        usage = agent.get_usage(mock_openai_result, result2)
        
        assert usage['requests'] == 2  # 1 + 1
        assert usage['request_tokens'] == 23  # 8 + 15
        assert usage['response_tokens'] == 9  # 3 + 6
        assert usage['thoughts_tokens'] == 4  # 1 + 3
        assert usage['total_tokens'] == 36  # 12 + 24


class TestCostCalculation:
    """Test new cost calculation functionality"""
    
    def test_openai_get_cost_single_result(self, mock_openai_result):
        """Test OpenAI get_cost with single result"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        
        costs = agent.get_cost(mock_openai_result)
        
        assert isinstance(costs, pd.Series)
        assert 'input_cost' in costs
        assert 'output_cost' in costs
        assert 'thoughts_cost' in costs
        assert 'total_cost' in costs
        
        # Verify cost calculation (gpt-4o: input=2.5, output=10.0, thoughts=10.0 per million)
        model_config = agent.allowed_models["gpt-4o"]
        expected_input = 8 * model_config.input_ppm_cost / 1_000_000  # 8 request tokens
        expected_output = 3 * model_config.output_ppm_cost / 1_000_000  # 3 response tokens
        expected_thoughts = 1 * model_config.thought_ppm_cost / 1_000_000  # 1 thoughts token
        expected_total = expected_input + expected_output + expected_thoughts
        
        assert abs(costs['input_cost'] - expected_input) < 1e-10
        assert abs(costs['output_cost'] - expected_output) < 1e-10
        assert abs(costs['thoughts_cost'] - expected_thoughts) < 1e-10
        assert abs(costs['total_cost'] - expected_total) < 1e-10
    
    def test_gemini_get_cost_single_result(self, mock_gemini_result):
        """Test Gemini get_cost with single result"""
        agent = GeminiAgent("Test", model_name="google-vertex:gemini-2.5-flash")
        
        costs = agent.get_cost(mock_gemini_result)
        
        assert isinstance(costs, pd.Series)
        
        # Verify cost calculation (gemini-2.5-flash: input=0.5, output=3.0, thoughts=3.0 per million)
        model_config = agent.allowed_models["google-vertex:gemini-2.5-flash"]
        expected_input = 12 * model_config.input_ppm_cost / 1_000_000  # 12 request tokens
        expected_output = 4 * model_config.output_ppm_cost / 1_000_000  # 4 response tokens
        expected_thoughts = 0 * model_config.thought_ppm_cost / 1_000_000  # 0 thoughts tokens
        expected_total = expected_input + expected_output + expected_thoughts
        
        assert abs(costs['input_cost'] - expected_input) < 1e-10
        assert abs(costs['output_cost'] - expected_output) < 1e-10
        assert abs(costs['thoughts_cost'] - expected_thoughts) < 1e-10
        assert abs(costs['total_cost'] - expected_total) < 1e-10
    
    def test_get_cost_multiple_results(self, mock_openai_result):
        """Test get_cost aggregates multiple results correctly"""
        agent = OpenAIAgent("Test", model_name="gpt-4o-mini")
        
        # Create a second result
        result2 = MockResult(output="8", usage_data={
            "requests": 1,
            "request_tokens": 20,
            "response_tokens": 10,
            "thoughts_tokens": 5,
            "total_tokens": 35
        })
        
        costs = agent.get_cost(mock_openai_result, result2)
        
        # Should aggregate: 8+20=28 request, 3+10=13 response, 1+5=6 thoughts
        model_config = agent.allowed_models["gpt-4o-mini"]
        expected_input = 28 * model_config.input_ppm_cost / 1_000_000
        expected_output = 13 * model_config.output_ppm_cost / 1_000_000
        expected_thoughts = 6 * model_config.thought_ppm_cost / 1_000_000
        expected_total = expected_input + expected_output + expected_thoughts
        
        assert abs(costs['input_cost'] - expected_input) < 1e-10
        assert abs(costs['output_cost'] - expected_output) < 1e-10
        assert abs(costs['thoughts_cost'] - expected_thoughts) < 1e-10
        assert abs(costs['total_cost'] - expected_total) < 1e-10
    
    def test_get_cost_no_results_raises_error(self):
        """Test get_cost raises ValueError when no results provided"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        
        with pytest.raises(ValueError, match="At least one result must be provided"):
            agent.get_cost()
    
    def test_get_cost_missing_model_raises_keyerror(self):
        """Test get_cost raises KeyError when model_name not in allowed_models"""
        agent = OpenAIAgent("Test", model_name="gpt-4o", validate_model_name=False)
        # Change model_name to something not in allowed_models
        agent.model_name = "nonexistent-model"
        
        mock_result = MockResult()
        
        with pytest.raises(KeyError, match="Model 'nonexistent-model' not found in allowed_models"):
            agent.get_cost(mock_result)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_token_usage(self):
        """Test cost calculation with zero tokens"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        
        zero_result = MockResult(usage_data={
            "requests": 1,
            "request_tokens": 0,
            "response_tokens": 0,
            "thoughts_tokens": 0,
            "total_tokens": 0
        })
        
        costs = agent.get_cost(zero_result)
        
        assert costs['input_cost'] == 0.0
        assert costs['output_cost'] == 0.0
        assert costs['thoughts_cost'] == 0.0
        assert costs['total_cost'] == 0.0
    
    def test_agent_without_model_name_attribute(self):
        """Test get_cost raises KeyError when agent has no model_name"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        delattr(agent, 'model_name')  # Remove model_name attribute
        
        mock_result = MockResult()
        
        with pytest.raises(KeyError, match="Model 'unknown' not found in allowed_models"):
            agent.get_cost(mock_result)
    
    def test_cost_calculation_accuracy(self):
        """Test that cost calculations are mathematically accurate"""
        agent = OpenAIAgent("Test", model_name="gpt-4o-mini")
        
        # gpt-4o-mini costs: input=0.15, output=0.6, thoughts=0.6 per million tokens
        model_config = agent.allowed_models["gpt-4o-mini"]
        
        # Test with specific token counts
        test_result = MockResult(usage_data={
            "requests": 1,
            "request_tokens": 1000,  # Exactly 1000 tokens for easy calculation
            "response_tokens": 500,   # Exactly 500 tokens
            "thoughts_tokens": 100,   # Exactly 100 tokens
            "total_tokens": 1600
        })
        
        costs = agent.get_cost(test_result)
        
        # Manual calculation
        expected_input = 1000 * 0.15 / 1_000_000  # 0.00015
        expected_output = 500 * 0.6 / 1_000_000   # 0.0003
        expected_thoughts = 100 * 0.6 / 1_000_000  # 0.00006
        expected_total = expected_input + expected_output + expected_thoughts  # 0.00051
        
        assert abs(costs['input_cost'] - expected_input) < 1e-10
        assert abs(costs['output_cost'] - expected_output) < 1e-10
        assert abs(costs['thoughts_cost'] - expected_thoughts) < 1e-10
        assert abs(costs['total_cost'] - expected_total) < 1e-10


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility"""
    
    def test_get_usage_unchanged_behavior(self):
        """Test that get_usage behavior is unchanged from before refactoring"""
        agent = OpenAIAgent("Test", model_name="gpt-4o")
        mock_result = MockResult()
        
        usage = agent.get_usage(mock_result)
        
        # Should return pandas Series with expected keys
        assert isinstance(usage, pd.Series)
        expected_keys = {'requests', 'request_tokens', 'response_tokens', 'thoughts_tokens', 'total_tokens'}
        assert set(usage.index) == expected_keys
        
        # Should have proper data types - but they may be numpy int64, so check they are numeric
        assert all(pd.api.types.is_numeric_dtype(type(usage[key])) for key in expected_keys)


class TestIntegration:
    """Integration tests for the full agent workflow"""
    
    def test_factory_to_cost_calculation_workflow(self):
        """Test complete workflow from factory creation to cost calculation"""
        # Create via factory
        agent = EZAgent("You are helpful", model_name="gpt-4o-mini")
        assert isinstance(agent, OpenAIAgent)
        
        # Mock a result and calculate costs
        mock_result = MockResult(usage_data={
            "requests": 1,
            "request_tokens": 100,
            "response_tokens": 50,
            "thoughts_tokens": 10,
            "total_tokens": 160
        })
        
        # Test usage tracking (backward compatibility)
        usage = agent.get_usage(mock_result)
        assert usage['requests'] == 1
        assert usage['total_tokens'] == 160
        
        # Test cost calculation (new functionality)
        costs = agent.get_cost(mock_result)
        assert costs['total_cost'] > 0
        assert costs['input_cost'] > 0
        assert costs['output_cost'] > 0
    
    def test_direct_instantiation_to_cost_workflow(self):
        """Test workflow with direct instantiation"""
        # Direct instantiation
        agent = GeminiAgent("You are helpful", model_name="google-vertex:gemini-2.5-pro")
        
        mock_result = MockResult(usage_data={
            "requests": 1,
            "request_tokens": 200,
            "response_tokens": 75,
            "thoughts_tokens": 0,
            "total_tokens": 275
        })
        
        usage = agent.get_usage(mock_result)
        costs = agent.get_cost(mock_result)
        
        # Should work the same as factory-created agents
        assert usage['total_tokens'] == 275
        assert costs['total_cost'] > 0
        assert costs['thoughts_cost'] == 0  # No thoughts tokens