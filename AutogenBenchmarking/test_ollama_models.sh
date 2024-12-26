#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test a single model with a test case
test_model() {
    local model=$1
    local test_name=$2
    local messages=$3
    local stream=$4

    echo -e "${BLUE}Testing $model - $test_name${NC}"
    echo "Request:"
    echo "$messages"
    echo
    echo "Response:"
    curl -s "http://localhost:11434/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model\",
            \"messages\": $messages,
            \"stream\": $stream,
            \"temperature\": 0.7,
            \"max_tokens\": 500
        }"
    echo -e "\n"
}

# Basic test cases
SYSTEM_MSG='{"role": "system", "content": "You are a helpful assistant."}'
MATH_MSG='{"role": "user", "content": "What is 2+2?"}'
CAPITAL_MSG='{"role": "user", "content": "What is the capital of France?"}'

# Test each model
echo -e "${GREEN}Starting Ollama model tests...${NC}"
echo "----------------------------------------"

for model in "wizardlm2" "nemotron-mini" "llama3.2"; do
    echo -e "${GREEN}Testing model: $model${NC}"
    echo "----------------------------------------"

    # Test basic math
    test_model "$model" "Basic Math" "[$SYSTEM_MSG, $MATH_MSG]" false

    # Test knowledge query
    test_model "$model" "Knowledge Query" "[$SYSTEM_MSG, $CAPITAL_MSG]" false

    echo "----------------------------------------"
done

echo -e "${GREEN}Testing streaming responses...${NC}"
echo "----------------------------------------"

# Test streaming with one model
test_model "wizardlm2" "Streaming Test" "[$SYSTEM_MSG, $MATH_MSG]" true

echo -e "${GREEN}All tests completed!${NC}"