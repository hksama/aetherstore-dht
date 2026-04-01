.PHONY: test test-fast test-slow test-all test-axioms test-matrix test-fuzzing test-intermediate test-coverage install-deps clean help

help:
	@echo "Reed-Solomon Testing Makefile"
	@echo "======================================"
	@echo "make test              - Run all tests (fast)"
	@echo "make test-all          - Run ALL tests including intermediate"
	@echo "make test-fast         - Run axiom + matrix tests (quick)"
	@echo "make test-slow         - Run full fuzzing suite (slow)"
	@echo "make test-axioms       - GF(256) axiom tests only"
	@echo "make test-matrix       - Matrix property tests only"
	@echo "make test-fuzzing      - Encoding fuzzing tests only"
	@echo "make test-intermediate - Test intermediate functions (invert, systematic, multiply)"
	@echo "make test-coverage     - Run tests with coverage report"
	@echo "make install-deps      - Install Python dependencies"
	@echo "make build             - Build Rust extension"
	@echo "make rebuild           - Full rebuild with maturin"
	@echo "make clean             - Clean build artifacts"

# Python interpreter
PYTHON := /Users/harshavardhankolhatkar04/Desktop/Projects/web3/aetherstore-dht/.venv/bin/python

# Test directories
TEST_DIR := python
PYTEST_OPTS := -v --tb=short

test: test-fast
	@echo "\n✅ All tests passed!"

test-all: test-axioms test-matrix test-intermediate test-fuzzing
	@echo "\n✅ All tests completed!"

test-fast: test-axioms test-matrix
	@echo "\n✅ Fast tests completed!"

test-slow: test-fuzzing test-intermediate
	@echo "\n✅ Fuzzing tests completed!"

test-axioms:
	@echo "Running GF(256) axiom tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_gf256_axioms.py $(PYTEST_OPTS)

test-matrix:
	@echo "Running Vandermonde matrix property tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_matrix_properties.py $(PYTEST_OPTS)

test-fuzzing:
	@echo "Running encoding fuzzing tests (this may take a minute)..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_encoding_fuzzing.py $(PYTEST_OPTS) -s

test-intermediate:
	@echo "Running intermediate function tests (invert, systematic, multiply)..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_intermediate_functions.py $(PYTEST_OPTS)

test-coverage:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest $(TEST_DIR)/ $(PYTEST_OPTS) --cov=reedsolomon --cov-report=html

install-deps:
	@echo "Installing Python dependencies..."
	pip install -r $(TEST_DIR)/requirements.txt

build:
	@echo "Building Rust extension with maturin..."
	PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 $(PYTHON) -m maturin develop

rebuild: clean build
	@echo "✅ Rebuild complete!"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	$(PYTHON) -m pytest --cache-clear
	cargo clean

# Run a specific test
test-one:
	@echo "Usage: make test-one TEST=test_gf256_axioms.py::TestGf256Closure::test_addition_closure"
	$(PYTHON) -m pytest $(TEST_DIR)/$(TEST) $(PYTEST_OPTS)

# Interactive Python shell with reedsolomon module loaded
interactive:
	@echo "Starting Python shell with reedsolomon module..."
	$(PYTHON) -c "import reedsolomon; print('reedsolomon module loaded'); import code; code.interact(local=locals())"

version:
	@$(PYTHON) -c "import reedsolomon; print('Reedsolomon version: 0.1.0')"

.DEFAULT_GOAL := help
