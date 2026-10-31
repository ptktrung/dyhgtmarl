# DyHGT-MARL. Thin wrapper over run_all.py so the usual steps are one word.
# Every target is safe to rerun: completed work is skipped.
#
#   make setup         install the dependencies
#   make data          build the integrated benchmark
#   make experiments   run experiments A to E plus the delay and artifact steps
#   make outputs       regenerate every table and figure
#   make paper         build the journal and IEEE Access documents
#   make verify        language report and delivery audit
#   make all           data, experiments, outputs, paper, verify
#   make kl            the Kuala Lumpur replication cohort
#   make stress        the revised stress regime of Section VI-G
#   make external DATA=<dir>   the METR-LA validation of Section VI-H
#   make package       the delivery archive
#   make clean         remove build intermediates, keep results

PY ?= python
DATA ?=

.PHONY: setup data experiments outputs paper verify all kl stress external \
        package clean list help

help list:
	@$(PY) run_all.py --list

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) run_all.py --stage data

experiments:
	$(PY) run_all.py --stage experiments

outputs:
	$(PY) run_all.py --stage outputs

paper:
	$(PY) run_all.py --stage paper

verify:
	$(PY) run_all.py --stage verify

all:
	$(PY) run_all.py

kl:
	$(PY) run_all.py --cohort kl

stress:
	$(PY) run_all.py --cohort stress

# METR-LA is not redistributed. DATA must point at a directory holding the
# three published files; docs/REPRODUCTION.md gives the download commands.
external:
	@test -n "$(DATA)" || { echo "usage: make external DATA=/path/to/metr-la"; exit 2; }
	$(PY) scripts/run_experiment_i_external.py --data "$(DATA)" --seeds 0 1 2 --epochs 8
	$(PY) scripts/tables/table_15_external_metr_la.py

package:
	$(PY) tools/package_delivery.py

clean:
	rm -rf build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
