#!/usr/bin/env bash
# Batch skill construction for workspace_state pipeline.
# Groups attempt directories by task ID, then runs the skill constructor
# for each task sequentially.
#
# Usage:
#   bash workspace_state/utils/batch_construct.sh <attempts_root> <output_root> [--from-scratch]
#
# Example:
#   bash workspace_state/utils/batch_construct.sh \
#     /space3/shuyu/project/skillDAG/harbor/skills-jobs/swebench-claude-5x6-mini \
#     /space3/shuyu/project/skillDAG/workspace_state/skill_lib

set -euo pipefail

ATTEMPTS_ROOT="${1:?Usage: $0 <attempts_root> <output_root> [--from-scratch]}"
OUTPUT_ROOT="${2:?Usage: $0 <attempts_root> <output_root> [--from-scratch]}"
EXTRA_ARGS="${3:-}"

# Collect unique task IDs
task_ids=()
for dir in "$ATTEMPTS_ROOT"/*/; do
    basename=$(basename "$dir")
    [[ -d "$dir" ]] || continue
    task_id="${basename%__*}"
    if [[ ! " ${task_ids[*]:-} " =~ " ${task_id} " ]]; then
        task_ids+=("$task_id")
    fi
done

echo "Found ${#task_ids[@]} tasks:"
printf "  %s\n" "${task_ids[@]}"
echo ""

for task_id in "${task_ids[@]}"; do
    echo "=========================================="
    echo "Constructing skill for: $task_id"
    echo "=========================================="

    attempt_dirs=()
    for dir in "$ATTEMPTS_ROOT"/"${task_id}"__*/; do
        [[ -d "$dir" ]] && attempt_dirs+=("${dir%/}")
    done

    output_path="$OUTPUT_ROOT/$task_id"

    echo "  Attempts: ${#attempt_dirs[@]}"
    echo "  Output:   $output_path"

    python -m workspace_state.agents.skill_constructor \
        "${attempt_dirs[@]}" \
        -o "$output_path" \
        $EXTRA_ARGS

    echo "  Done."
    echo ""
done

echo "All tasks completed."
