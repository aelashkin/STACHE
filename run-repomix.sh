#!/usr/bin/env bash
set -euo pipefail

# run-repomix.sh
# OUTPUT: AI-friendly single file pack of the entire project codebase
# DESCRIPTION: Runs repomix while ignoring data and other files we don't want
# included in repomix's output. For convenience it forwards extra arguments
# to repomix.

SCRIPT_NAME="$(basename "$0")"

# Patterns to ignore (comma-separated when passed to repomix)
IGNORE_PATTERNS=(
    '**/*.png'
    '**/*.svg'
    "${SCRIPT_NAME}"
    'LICENCE'
    '__pycache__/**'
    '.venv/**'
    '.env'
    'poetry.lock'
    'data/**'
    'tests/**'
    'docs/**'
    'assets/**'
)

# Join patterns with commas
IFS=','
IGNORE_ARG="${IGNORE_PATTERNS[*]}"
unset IFS

echo "Running: repomix --ignore \"${IGNORE_ARG}\" $*"

exec repomix --ignore "${IGNORE_ARG}" "$@"
