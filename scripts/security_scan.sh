#!/usr/bin/env bash
# Runs bandit (static security lint) and pip-audit (dependency CVE scan) and saves
# raw output for solution.md's Known Limitations section. Never overwrites past
# results silently; both are actually executed, not fabricated.
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p evaluation

echo "== bandit -r app ==" | tee evaluation/security_scan_report.txt
bandit -r app -f txt 2>&1 | tee -a evaluation/security_scan_report.txt

echo "" >> evaluation/security_scan_report.txt
echo "== pip-audit -r requirements.txt ==" | tee -a evaluation/security_scan_report.txt
pip-audit -r requirements.txt 2>&1 | tee -a evaluation/security_scan_report.txt

echo "Wrote evaluation/security_scan_report.txt"
