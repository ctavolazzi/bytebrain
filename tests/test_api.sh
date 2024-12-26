#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Base URL
BASE_URL="http://localhost:8000/api/benchmarks"

# Test function
test_endpoint() {
    local description=$1
    local command=$2
    echo -e "\n${GREEN}Testing: $description${NC}"
    eval $command
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
    fi
}

# Basic conversation config
BASIC_CONFIG='{
    "prompt": "What is the capital of France?",
    "configs": [{
        "name": "basic_test",
        "agents": [
            {
                "name": "assistant",
                "type": "assistant",
                "llm_config": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "api_key": "'$OPENAI_API_KEY'"
                },
                "system_message": "You are a helpful AI assistant."
            },
            {
                "name": "user",
                "type": "user_proxy",
                "system_message": "You are a user seeking assistance."
            }
        ],
        "initiator": "user",
        "max_rounds": 3,
        "description": "Basic test conversation"
    }],
    "parallel_processing": false
}'

# Multi-agent config
MULTI_AGENT_CONFIG='{
    "prompt": "Design and implement a Python function to calculate Fibonacci numbers.",
    "configs": [{
        "name": "multi_agent_test",
        "agents": [
            {
                "name": "planner",
                "type": "assistant",
                "llm_config": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": "'$OPENAI_API_KEY'"
                },
                "system_message": "You are a planning assistant that breaks down tasks."
            },
            {
                "name": "coder",
                "type": "assistant",
                "llm_config": {
                    "provider": "anthropic",
                    "model": "claude-2",
                    "api_key": "'$ANTHROPIC_API_KEY'"
                },
                "system_message": "You are a coding assistant that implements solutions."
            },
            {
                "name": "user",
                "type": "user_proxy",
                "system_message": "You are a user seeking assistance."
            }
        ],
        "initiator": "user",
        "max_rounds": 5,
        "description": "Multi-agent test conversation"
    }],
    "parallel_processing": true
}'

# Test endpoints
echo -e "${GREEN}Starting API tests...${NC}"

# Test basic conversation benchmark
test_endpoint "Basic conversation benchmark" "curl -s -X POST $BASE_URL/run \
    -H 'Content-Type: application/json' \
    -d '$BASIC_CONFIG' | jq"

# Test multi-agent benchmark
test_endpoint "Multi-agent benchmark" "curl -s -X POST $BASE_URL/run \
    -H 'Content-Type: application/json' \
    -d '$MULTI_AGENT_CONFIG' | jq"

# Test benchmark history
test_endpoint "Get benchmark history" "curl -s $BASE_URL/history | jq"

# Get the most recent benchmark ID
BENCHMARK_ID=$(curl -s $BASE_URL/history | jq -r '.[0].id')

# Test specific benchmark retrieval
test_endpoint "Get specific benchmark" "curl -s $BASE_URL/history/$BENCHMARK_ID | jq"

# Test streaming updates (requires separate terminal for viewing)
echo -e "\n${GREEN}Testing streaming updates...${NC}"
echo "To test streaming updates, run this command in another terminal:"
echo "curl -N $BASE_URL/stream"