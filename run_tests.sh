#!/bin/bash
# Test runner script for qex

set -e

echo "Setting up test environment..."
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not in a virtual environment. Consider activating one."
    echo ""
fi

# Install/update qex in development mode
echo "Installing qex in development mode..."
pip install -e . > /dev/null 2>&1 || {
    echo "Failed to install qex. Make sure dependencies are installed:"
    echo "  pip install cirq numpy"
    exit 1
}

echo "Running qex tests..."
echo ""

# Run the test script
python test_qex.py

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
