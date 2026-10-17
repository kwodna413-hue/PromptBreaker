# PromptBreaker
PromptBreaker

A lightweight Python tool for testing LLM prompt-injection resistance and instruction-following behavior.

Features

- Document prompt-injection tests
- Instruction and formatting conflict tests
- Quoted-instruction translation tests
- JSONL results and Markdown reports
- Configurable API endpoint and model

Requirements

- Python 3.10+
- requests

Configuration

Set these environment variables before running:

- "PB_API_KEY": your API key
- "PB_BASE_URL": API base URL
- "PB_MODEL": model name

Usage

pip install requests
python prompt_breaker.py

Limitations

This is an experimental prototype. Automated verdicts are preliminary and require manual review. The included tests are illustrative and do not provide a comprehensive security evaluation.

Never publish API keys, credentials, or private test results.

Status

Early prototype. Contributions and issue reports are welcome.
