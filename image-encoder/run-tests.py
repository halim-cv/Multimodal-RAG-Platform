#!/usr/bin/env python3
"""
Test runner script
Usage: python run-tests.py
"""
import sys
import pytest

if __name__ == "__main__":
    # Run pytest with verbose output
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    sys.exit(exit_code)
