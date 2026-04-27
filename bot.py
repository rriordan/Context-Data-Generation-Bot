#!/usr/bin/env python3
"""Context Data Generation Bot — interactive CLI for building personalized LLM context snippets.

Uses the `claude` CLI for all model calls. Profiles are persisted under profiles/<name>/.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.theme import Theme

# ── Question pool ─────────────────────────────────────────────────────────────
# Ordered by how much the context is likely to improve AI responses.
# Sections progress from high-signal fundamentals to niche/personal details.

@dataclass
class Question:
    id: str
    section: str
    text: str


QUESTIONS: list[Question] = [
    # ── Identity ── highest signal; used in nearly every interaction
    Question("full_name",       "Identity",       "What is your full name?"),
    Question("location",        "Identity",       "Where do you currently live? (city and country)"),
    Question("occupation",      "Identity",       "What is your occupation or job title?"),
    Question("industry",        "Identity",       "What industry or field do you work in?"),
    Question("languages",       "Identity",       "What languages do you speak, and at what level of fluency?"),
    Question("nationality",     "Identity",       "What is your nationality or cultural background?"),
    Question("age_range",       "Identity",       "What is your age or approximate age range?"),

    # ── AI interaction preferences ── directly shape every response
    Question("response_format", "AI Preferences", "How do you prefer AI responses to be formatted? "
                                                   "(e.g. bullet points, prose, tables, code blocks)"),
    Question("response_length", "AI Preferences", "Do you prefer concise answers or thorough explanations?"),
    Question("tone",            "AI Preferences", "What tone do you prefer from an AI assistant? "
                                                   "(e.g. formal, casual, direct, friendly)"),
    Question("technical_depth", "AI Preferences", "What level of technical depth do you prefer? "
                                                   "(e.g. plain English, intermediate, expert-level)"),
    Question("expert_areas",    "AI Preferences", "What subjects are you already expert in, so an AI "
                                                   "can skip basic explanations?"),
    Question("ai_use_cases",    "AI Preferences", "What tasks do you most often use AI assistants for?"),

    # ── Professional ── high signal for work-related queries
    Question("top_skills",      "Professional",   "What are your most important professional skills?"),
    Question("career_goals",    "Professional",   "What are your main career goals or ambitions?"),
    Question("education",       "Professional",   "Describe your educational background."),
    Question("tools_tech",      "Professional",   "What software, tools, or technologies do you use regularly?"),
    Question("work_challenges", "Professional",   "What are your biggest professional challenges right now?"),
    Question("current_projects","Professional",   "What kind of projects or work are you currently focused on?"),
    Question("work_style",      "Professional",   "How would colleagues describe your working style?"),

    # ── Personal context ── useful for lifestyle and general queries
    Question("hobbies",         "Personal",       "What are your main hobbies and interests?"),
    Question("family",          "Personal",       "Describe your family or household situation."),
    Question("health",          "Personal",       "Do you have any health considerations, dietary restrictions, "
                                                   "or physical limitations worth knowing?"),
    Question("living_situation","Personal",       "Describe your living situation. "
                                                   "(e.g. own/rent, house/flat, city/rural)"),
    Question("social_energy",   "Personal",       "Are you more introverted or extroverted, "
                                                   "and what socially energises or drains you?"),

    # ── Preferences ──
    Question("music",           "Preferences",    "Describe your taste in music."),
    Question("books_films",     "Preferences",    "What genres of books, films, or TV shows do you enjoy?"),
    Question("travel",          "Preferences",    "What are your favourite or dream travel destinations?"),
    Question("food",            "Preferences",    "What are your food preferences, favourite cuisines, "
                                                   "or notable dislikes?"),
    Question("sport_fitness",   "Preferences",    "What sports, fitness activities, or outdoor pursuits "
                                                   "do you enjoy?"),

    # ── Values & worldview ──
    Question("core_values",     "Values",         "What are your core personal values?"),
    Question("causes",          "Values",         "What social, environmental, or political causes "
                                                   "matter most to you?"),
    Question("philosophy",      "Values",         "Describe your philosophical, religious, or spiritual outlook."),
    Question("political_views", "Values",         "How would you describe your political views? "
                                                   "(optional — skip if preferred)"),

    # ── Quirks & specifics ── lower signal but adds colour
    Question("unusual_skills",  "Quirks",         "What is an unusual skill or hobby most people "
                                                   "don't know you have?"),
    Question("strong_opinions", "Quirks",         "What is a strong opinion you hold that might "
                                                   "surprise people?"),
    Question("pet_peeves",      "Quirks",         "What are your biggest pet peeves?"),
    Question("niche_interests", "Quirks",         "What niche topics can you talk about endlessly?"),
    Question("motivators",      "Quirks",         "What motivates or energises you most?"),
    Question("ideal_day",       "Quirks",         "Describe your ideal day."),
    Question("ai_wish",         "Quirks",         "What recurring problem in your life do you "
                                                   "most wish an AI could help with?"),
    Question("deep_knowledge",  "Quirks",         "What topic do you know surprisingly well "
                                                   "compared to most people?"),
    Question("fictional_ident", "Quirks",         "Which fictional character do you most identify with, "
                                                   "and why?"),
    Question("thinking_style",  "Quirks",         "What is the most important thing to understand "
                                                   "about how you think or make decisions?"),
]

SECTIONS = list(dict.fromkeys(q.section for q in QUESTIONS))  # ordered, deduped

# ── Profile ───────────────────────────────────────────────────────────────────

PROFILES_DIR = Path("profiles")
LAST_USED_FILE = PROFILES_DIR / "last_used.txt"


@dataclass
class Profile:
    name: str
    answered: dict[str, str] = field(default_factory=dict)   # q.id → snippet filename
    skipped: list[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return PROFILES_DIR / _safe_name(self.name)

    @property
    def json_path(self) -> Path:
        return self.dir / "profile.json"

    @property
    def snippets_dir(self) -> Path:
        return self.dir / "snippets"

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snippets_dir.mkdir(exist_ok=True)
        self.json_path.write_text(
            json.dumps({"name": self.name, "answered": self.answered, "skipped": self.skipped},
                       indent=2),
            encoding="utf-8",
        )
        LAST_USED_FILE.write_text(self.name, encoding="utf-8")

    @classmethod
    def load(cls, name: str) -> "Profile":
        path = PROFILES_DIR / _safe_name(name) / "profile.json"
        if not path.exists():
            return cls(name=name)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(name=data["name"], answered=data.get("answered", {}),
                   skipped=data.get("skipped", []))

    @classmethod
    def last_used_name(cls) -> str | None:
        if LAST_USED_FILE.exists():
            name = LAST_USED_FILE.read_text(encoding="utf-8").strip()
            if name and (PROFILES_DIR / _safe_name(name)).exists():
                return name
        return None


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.lower())


# ── Claude CLI ────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-7"

FORMATTER_SYSTEM = """\
You are a context data formatting tool. Transform raw user input into clean, structured \
contextual snippets for a vector database that will ground an LLM.

You will receive the user's name, the question asked, and their raw response.

Rules:
- Extract only persistent facts (not ephemeral things like today's weather)
- Rewrite everything in the third person using the user's name
- Group related facts under markdown headings
- Remove filler words, repetition, and irrelevant content
- Return ONLY the formatted snippet inside a markdown code fence — no commentary outside it

Example (name: Daniel, question: dream travel destinations):
```markdown
## Travel Preferences

### Dream Destinations
Daniel dreams of visiting Japan for its culture and cuisine. He is also drawn to Iceland \
for its landscapes and hot springs.
```"""

console = Console(theme=Theme({"success": "green", "info": "cyan", "warn": "yellow"}))


def _claude(system_prompt: str, user_message: str) -> str:
    result = subprocess.run(
        ["claude", "--print", "--model", MODEL,
         "--system-prompt", system_prompt,
         "--tools", "", "--output-format", "text",
         user_message],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        console.print(f"[warn]claude error (exit {result.returncode}):[/warn] {err}")
        sys.exit(1)
    return result.stdout.strip()


def format_context(user_name: str, question: str, user_input: str) -> str:
    return _claude(
        FORMATTER_SYSTEM,
        f"User's name: {user_name}\nQuestion: {question}\nResponse: {user_input}\n\n"
        "Format this into a contextual snippet.",
    )


def extract_snippet(text: str) -> str:
    m = re.search(r"```(?:markdown)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


# ── UI helpers ────────────────────────────────────────────────────────────────

def show_progress(profile: Profile) -> None:
    table = Table(title=f"Context library — {profile.name}", show_header=True,
                  header_style="bold cyan")
    table.add_column("Section")
    table.add_column("Done", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("Remaining", justify="right")

    total_done = total_skipped = total_remaining = 0
    for section in SECTIONS:
        qs = [q for q in QUESTIONS if q.section == section]
        done = sum(1 for q in qs if q.id in profile.answered)
        skipped = sum(1 for q in qs if q.id in profile.skipped and q.id not in profile.answered)
        remaining = len(qs) - done - skipped
        table.add_row(section, str(done), str(skipped) if skipped else "—", str(remaining))
        total_done += done; total_skipped += skipped; total_remaining += remaining

    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total_done}[/bold]",
                  f"[bold]{total_skipped}[/bold]" if total_skipped else "—",
                  f"[bold]{total_remaining}[/bold]")
    console.print(table)


def collect_multiline_input(prompt_text: str = "") -> str:
    if prompt_text:
        console.print(f"[info]{prompt_text}[/info]")
    console.print("[dim]Press Enter on a blank line when done.[/dim]")
    lines: list[str] = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def next_question(profile: Profile) -> Question | None:
    """Return the next unanswered, non-skipped question, or None if all done."""
    done_or_skipped = set(profile.answered) | set(profile.skipped)
    for q in QUESTIONS:
        if q.id not in done_or_skipped:
            return q
    return None


def pick_answered_question(profile: Profile) -> Question | None:
    """Let the user pick a previously answered question to re-answer."""
    answered_qs = [q for q in QUESTIONS if q.id in profile.answered]
    if not answered_qs:
        console.print("[warn]No answered questions yet.[/warn]")
        return None

    console.print("\n[info]Answered questions:[/info]")
    for i, q in enumerate(answered_qs, 1):
        console.print(f"  [{i}] ({q.section}) {q.text}")
    console.print("  [0] Cancel")

    while True:
        raw = Prompt.ask("Pick a number").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(answered_qs):
            return answered_qs[int(raw) - 1]
        console.print("[warn]Invalid choice.[/warn]")


# ── Session ───────────────────────────────────────────────────────────────────

def answer_question(profile: Profile, q: Question, is_edit: bool = False) -> None:
    """Present a question, collect the answer, format it, and save."""
    label = "Re-answering" if is_edit else "Question"
    console.print(Panel(
        f"[bold]{q.text}[/bold]\n[dim]Section: {q.section}[/dim]",
        title=f"[cyan]{label}[/cyan]",
        border_style="cyan",
    ))

    user_input = collect_multiline_input()
    if not user_input:
        console.print("[warn]No input — skipping.[/warn]")
        return

    console.print("\n[info]Formatting snippet…[/info]")
    formatted = format_context(profile.name, q.text, user_input)
    snippet_content = extract_snippet(formatted)

    console.print(Panel(Markdown(snippet_content), title="[green]Snippet[/green]",
                        border_style="bright_blue"))

    if not Confirm.ask("\n[info]Save this snippet?[/info]", default=True):
        return

    # Save file
    profile.snippets_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{q.id}.md"
    filepath = profile.snippets_dir / filename
    header = (f"---\ngenerated: {datetime.now().isoformat()}\n"
              f"question_id: {q.id}\nquestion: {q.text}\n---\n\n")
    filepath.write_text(header + snippet_content + "\n", encoding="utf-8")

    # Remove old file if re-answering
    if is_edit and q.id in profile.answered:
        old_file = profile.snippets_dir / profile.answered[q.id]
        if old_file.exists():
            old_file.unlink()

    profile.answered[q.id] = filename
    profile.skipped = [s for s in profile.skipped if s != q.id]  # un-skip if was skipped
    profile.save()
    console.print(f"[success]Saved → {filepath}[/success]")


def run_session(profile: Profile) -> None:
    """Main question loop for a session."""
    show_progress(profile)

    while True:
        q = next_question(profile)

        if q is None:
            console.print(
                "\n[success]All questions answered! "
                "You can still edit previous answers.[/success]"
            )
            if not Confirm.ask("[info]Edit a previous answer?[/info]", default=False):
                break
            q = pick_answered_question(profile)
            if q:
                answer_question(profile, q, is_edit=True)
            continue

        console.print(Panel(
            f"[bold]{q.text}[/bold]\n[dim]Section: {q.section}[/dim]",
            title="[cyan]Next Question[/cyan]",
            border_style="cyan",
        ))

        action = Prompt.ask(
            "\n[info]\\[a]nswer  \\[s]kip  \\[e]dit previous  \\[q]uit[/info]",
            default="a",
        ).strip().lower()

        if action == "q":
            break
        elif action == "s":
            if q.id not in profile.skipped:
                profile.skipped.append(q.id)
                profile.save()
            console.print("[info]Skipped.[/info]")
        elif action == "e":
            eq = pick_answered_question(profile)
            if eq:
                answer_question(profile, eq, is_edit=True)
        else:  # "a" or anything else
            answer_question(profile, q)


# ── Startup ───────────────────────────────────────────────────────────────────

def load_or_create_profile() -> Profile:
    last = Profile.last_used_name()
    if last:
        if Confirm.ask(f"\n[info]Load profile for [bold]{last}[/bold]?[/info]", default=True):
            profile = Profile.load(last)
            console.print(f"[success]Loaded profile: {profile.name}[/success]")
            return profile

    name = Prompt.ask("\n[info]Enter your name[/info]").strip() or "User"
    profile = Profile.load(name)  # loads existing or creates empty
    profile.dir.mkdir(parents=True, exist_ok=True)
    profile.snippets_dir.mkdir(exist_ok=True)
    profile.save()
    console.print(f"[success]Profile ready: {profile.dir}[/success]")
    return profile


def main() -> None:
    console.print(Panel(
        "[bold]Context Data Generation Bot[/bold]\n"
        "[dim]Build a personalized context library for your LLM[/dim]",
        border_style="bright_blue",
    ))

    if subprocess.run(["claude", "--version"], capture_output=True).returncode != 0:
        console.print(
            "[warn]The `claude` CLI was not found. "
            "Install it from https://claude.ai/code and try again.[/warn]"
        )
        sys.exit(1)

    profile = load_or_create_profile()
    run_session(profile)

    total = len(profile.answered)
    console.print(
        f"\n[success]Session complete. "
        f"{total} snippet{'s' if total != 1 else ''} in {profile.snippets_dir}[/success]"
    )


if __name__ == "__main__":
    main()
