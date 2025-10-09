# Agent implementations

* `default.py` - Minimal default agent implementation.
* `interactive.py` - Extends `default.py` with some minimal human-in-the-loop functionality (confirm actions, etc.).
* `interactive_textual.py` - Extends `default.py` with [Textual](https://textual.textualize.io/) for an interactive TUI.
   (this is a more complicated UI).
* `summarize_with_workflow.py` - Extends `default.py` with intelligent workflow-based message compression.
   Uses a summary model to condense conversation history while preserving critical context.