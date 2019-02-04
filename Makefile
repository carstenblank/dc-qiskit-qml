PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3
COVERAGE := --cov=dc_qiskit_qml --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest tests

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install dc-qiskit-qml"
	@echo "  wheel              to build the dc-qiskit-qml wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite for all configured devices"
	@echo "  test-[device]      to run the test suite for the device simulator or ibm"
	@echo "  coverage           to generate a coverage report for all configured devices"
	@echo "  coverage-[device]  to generate a coverage report for the device simulator or ibm"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install dc-qiskit-qml you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist

.PHONY : clean
clean:
	rm -rf dc_qiskit_qml/__pycache__
	rm -rf tests/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache
	rm -rf .coverage coverage_html_report/

docs:
	make -C docs html

.PHONY : clean-docs
clean-docs:
	make -C docs clean

test:
	$(PYTHON) $(TESTRUNNER)

coverage:
	@echo "Generating coverage report..."
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)
