## nl2lumos

This directory contains the natural-language-to-Lumos tooling for the project.

- **`nlspecs/`**: Natural language specifications written as Jinja templates (`*.jinja`).
- **`spec_progs_claude/`**: Lumos DSL programs generated from specs via Claude (`*.lumos`).
- **`py_specs_claude/`**: Python reference implementations generated from specs via Claude (`*.py`).

### Main scripts

- **`call_claude.py`**: Calls Bedrock Claude with `claude_prompt.jinja` to generate Lumos programs.
  - All specs: `python -m nl2lumos.call_claude`
  - Single spec: `python -m nl2lumos.call_claude nl2lumos/nlspecs/your_spec.jinja`

- **`call_claude_python.py`**: Calls Bedrock Claude with `claude_prompt_py.jinja` to generate Python programs.
  - All specs: `python -m nl2lumos.call_claude_python`
  - Single spec: `python -m nl2lumos.call_claude_python nl2lumos/nlspecs/your_spec.jinja`

- **`compare_loc.py`**: Compares non-comment lines of code between generated Python and Lumos programs.
  - `python nl2lumos/compare_loc.py`

