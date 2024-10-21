#!/bin/bash
# Make a graph of the dependencies to get to a specific task
#
# Usage:
# bash scripts/make-graph-for-task.sh <task> <output-file-name>
# e.g.
# bash scripts/make-graph-for-task.sh '(40yy_write-input4mips/4001_write-input4mips-files) write input4MIPs - write all files:so2f2' so2f2.pdf
#
# In practice, you will generally need to run this within our pixi environment i.e.
# pixi run bash scripts/make-graph-for-task.sh <task> <output-file-name>
#
#
# To see a list of all the available tasks, use
# `pixi run doit list`
# The tasks are then the left-hand column of the output
# (i.e. everything except the task descriptions).

task=$1
output_file_name=$2
dot_name="${output_file_name}.dot"

echo "Creating dependency graph for ${task}"

doit graph -o "${dot_name}" --reverse --show-subtasks "${task}"

echo "Saving output to ${output_file_name}"
# Just in case, make sure `dot` is configured
dot -c
# dot -Tpng "${dot_name}" -o "${output_file_name}"
dot -Tpdf "${dot_name}" -o "${output_file_name}"
