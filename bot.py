#!/usr/bin/env python3
"""Context Data Generation Bot — interactive CLI for building personalized LLM context snippets.

Uses the `claude` CLI for all model calls rather than the Anthropic SDK directly.
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.theme import Theme

# ── System prompts (drawn from configurations/) ──────────────────────────────

TOPIC_SYSTEM_PROMPT = """\
Your purpose is to act as a friendly assistant helping the user generate a library \
of contextual data to enhance the capabilities of a large language model tool.

The user's intention is to provide a broad store of contextual data which can be used \
for the purpose of making the LLM's outputs more targeted and personalized.

Your function is to suggest a single random topic for a piece of contextual data that \
the user should generate.

The user will specify an obscurity level from 1 to 5:
- Level 1: Very basic, not obscure (e.g., city of birth, nationality)
- Level 2: Somewhat basic (e.g., career aspirations, dream travel destinations, music taste)
- Level 3: Medium obscurity (e.g., unexpected skills, hobbies most people wouldn't guess)
- Level 4: Quite obscure (e.g., niche interests, unusual food combinations, favourite \
fictional characters)
- Level 5: Highly obscure (e.g., intricate dreams, bizarre connections between topics, \
hypothetical inventions)

When suggesting topics, try to suggest ones that will yield a few pieces of information \
together rather than just one data point. Exception: levels 4-5 may focus on a single \
very obscure data point.

Respond with ONLY the topic question — no preamble, no explanation, just the question itself.\
"""

FORMATTER_SYSTEM_PROMPT = """\
You are a context data formatting tool. Your purpose is to transform raw user input into \
clean, structured contextual snippets suitable for storage in a vector database.

You will receive:
1. The user's name
2. The topic/question that was suggested
3. The user's raw response (which may be informal, include speech artefacts, or be unstructured)

Your task:
- Extract only persistent, useful contextual information (not ephemeral facts like today's weather)
- Rewrite all information in the third person using the user's name
- Organise information under appropriate markdown headings
- Group similar pieces of information together
- Remove filler words, redundancies, and irrelevant content
- Return ONLY the formatted snippet inside a markdown code fence, with no additional commentary \
before or after it

Example — if the user's name is "Daniel" and they say \
"um I've always wanted to visit Japan for the culture and food, oh and Iceland too":

```markdown
## Travel Preferences

### Dream Destinations
Daniel dreams of visiting Japan, drawn by its culture and cuisine. He is also interested \
in traveling to Iceland.
```\
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

console = Console(theme=Theme({"success": "green", "info": "cyan", "warn": "yellow"}))

MODEL = "claude-opus-4-7"


def _claude(system_prompt: str, user_message: str) -> str:
    """Call the `claude` CLI in non-interactive mode and return the response text."""
    result = subprocess.run(
        [
            "claude",
            "--print",
            "--model", MODEL,
            "--system-prompt", system_prompt,
            "--tools", "",          # disable all tools — pure text generation
            "--output-format", "text",
            user_message,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        console.print(f"[warn]claude CLI error (exit {result.returncode}):[/warn] {err}")
        sys.exit(1)
    return result.stdout.strip()


def suggest_topic(obscurity_level: int) -> str:
    """Ask Claude to suggest a topic at the given obscurity level."""
    return _claude(
        TOPIC_SYSTEM_PROMPT,
        f"Suggest a topic at obscurity level {obscurity_level}.",
    )


def format_context(user_name: str, topic: str, user_input: str) -> str:
    """Format raw user input into a structured context snippet."""
    return _claude(
        FORMATTER_SYSTEM_PROMPT,
        (
            f"User's name: {user_name}\n"
            f"Topic/Question: {topic}\n"
            f"User's response: {user_input}\n\n"
            "Please format this into a contextual snippet."
        ),
    )


def extract_snippet(formatted_response: str) -> str:
    """Pull markdown content out of a code fence if present."""
    match = re.search(r"```(?:markdown)?\n(.*?)```", formatted_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return formatted_response


def save_snippet(snippet_content: str, user_name: str, topic: str) -> Path:
    """Save a snippet as a markdown file under output/."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r"[^\w\s-]", "", topic[:40]).strip().replace(" ", "_").lower()
    filename = f"{timestamp}_{safe_topic}.md"
    filepath = output_dir / filename

    header = (
        f"---\n"
        f"generated: {datetime.now().isoformat()}\n"
        f"subject: {user_name}\n"
        f"topic: {topic}\n"
        f"---\n\n"
    )
    filepath.write_text(header + snippet_content + "\n", encoding="utf-8")
    return filepath


def collect_multiline_input() -> str:
    """Read multi-line input until the user submits a blank line."""
    console.print(
        "[info]Share your thoughts below. Press Enter on a blank line when done.[/info]"
    )
    lines: list[str] = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def prompt_obscurity() -> int:
    """Display obscurity descriptions and return the chosen level (1–5)."""
    console.print("\n[info]Obscurity levels:[/info]")
    descriptions = [
        "1 – Basic          (city of birth, nationality)",
        "2 – Common         (career aspirations, music taste)",
        "3 – Moderate       (unexpected skills, hidden hobbies)",
        "4 – Obscure        (niche interests, unusual preferences)",
        "5 – Very obscure   (hypothetical scenarios, bizarre connections)",
    ]
    for d in descriptions:
        console.print(f"  {d}")

    while True:
        raw = Prompt.ask("\n[info]Choose obscurity level[/info]", default="3")
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        console.print("[warn]Please enter a number between 1 and 5.[/warn]")


# ── Session loop ──────────────────────────────────────────────────────────────

def run_session(user_name: str) -> None:
    """Run one full cycle: topic → input → format → (optionally) save."""
    obscurity = prompt_obscurity()

    console.print("\n[info]Generating topic…[/info]")
    topic = suggest_topic(obscurity)
    console.print(Panel(topic, title="[green]Suggested Topic[/green]", border_style="green"))

    user_input = collect_multiline_input()
    if not user_input:
        console.print("[warn]No input provided — skipping.[/warn]")
        return

    console.print("\n[info]Formatting your context snippet…[/info]")
    formatted = format_context(user_name, topic, user_input)
    snippet_content = extract_snippet(formatted)

    console.print(
        Panel(
            Markdown(snippet_content),
            title="[green]Context Snippet[/green]",
            border_style="bright_blue",
        )
    )

    if Confirm.ask("\n[info]Save this snippet to file?[/info]", default=True):
        filepath = save_snippet(snippet_content, user_name, topic)
        console.print(f"[success]Saved → {filepath}[/success]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    console.print(
        Panel(
            "[bold]Context Data Generation Bot[/bold]\n"
            "[dim]Build a personalized context library for your LLM[/dim]",
            border_style="bright_blue",
        )
    )

    # Verify the claude CLI is available
    if subprocess.run(["claude", "--version"], capture_output=True).returncode != 0:
        console.print(
            "[warn]The `claude` CLI was not found. "
            "Install it from https://claude.ai/code and try again.[/warn]"
        )
        sys.exit(1)

    user_name = Prompt.ask("\n[info]What's your name?[/info]").strip() or "User"
    console.print(f"\n[success]Hello, {user_name}! Let's build your context library.[/success]")

    while True:
        run_session(user_name)
        if not Confirm.ask("\n[info]Generate another snippet?[/info]", default=True):
            break

    snippet_count = len(list(Path("output").glob("*.md"))) if Path("output").exists() else 0
    console.print(
        f"\n[success]Session complete!"
        f"{' ' + str(snippet_count) + ' snippet(s) saved in output/' if snippet_count else ''}[/success]"
    )


if __name__ == "__main__":
    main()
