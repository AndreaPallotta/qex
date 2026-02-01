#!/bin/bash
# Test runner script for qlab

set -e

echo "Setting up test environment..."
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not in a virtual environment. Consider activating one."
    echo ""
fi

# Install/update qlab in development mode
echo "Installing qlab in development mode..."
pip install -e . > /dev/null 2>&1 || {
    echo "Failed to install qlab. Make sure dependencies are installed:"
    echo "  pip install cirq numpy"
    exit 1
}

echo "Running qlab tests..."
echo ""

# Run the test script
python test_qlab.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All tests completed successfully!"
    exit 0
else
    echo ""
    echo "✗ Some tests failed"
    exit 1
fi
