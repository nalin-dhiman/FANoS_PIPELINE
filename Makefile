\
PY=python

.PHONY: all ingest experiments validate package clean

all:
	$(PY) reproduce/build_all.py

ingest:
	@echo "Ingest mode: assumes raw CSVs already exist in artifacts/raw"

experiments:
	@echo "NOTE: Only quadratic sweep + diagnostics are auto-generated in this minimal build."
	@echo "To fully re-run Rosenbrock, ablations, PINNs, implement/run the corresponding scripts."

validate:
	$(PY) reproduce/validate_outputs.py

package:
	$(PY) reproduce/package_zip.py

clean:
	rm -rf artifacts/standardized artifacts/stats artifacts/plots artifacts/tables latex_snippets
	mkdir -p artifacts/raw artifacts/standardized artifacts/stats artifacts/plots artifacts/tables latex_snippets
