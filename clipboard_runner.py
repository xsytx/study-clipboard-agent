"""Desktop Study & Clipboard Agent. Run: python clipboard_runner.py."""

import os
import queue
import re
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import get_llm, load_environment
from db import init_db
from tools import ALL_TOOLS

try:
    import pyperclip
except ImportError:
    pyperclip = None


SYSTEM_PROMPT = (
    "You are Study & Clipboard Agent. You have these tools: fix_layout, "
    "detect_language, translate_text, explain_selection, grammar_check, "
    "make_flashcards, save_note, search_notes, convert_currency. "
    "Use them when relevant. Be concise."
)


def normalize_agent_output(result: dict) -> dict:
    """Keep final content blocks inside one answer before saving chat history."""
    output = result["output"]
    if isinstance(output, (list, BaseMessage)):
        message = output if isinstance(output, BaseMessage) else AIMessage(content=output)
        return {**result, "output": StrOutputParser().invoke(message)}
    return result


def build_agent() -> RunnableWithMessageHistory:
    """Build the notebook's agent stack with in-memory history for this run."""
    load_environment()
    init_db()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(get_llm(), ALL_TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent, tools=ALL_TOOLS, verbose=False,
        return_intermediate_steps=True, handle_parsing_errors=True,
        max_iterations=6, max_execution_time=180,
    )
    histories: dict[str, ChatMessageHistory] = {}

    def get_history(session_id: str) -> ChatMessageHistory:
        if session_id not in histories:
            histories[session_id] = ChatMessageHistory()
        return histories[session_id]

    return RunnableWithMessageHistory(
        executor | RunnableLambda(normalize_agent_output), get_history,
        input_messages_key="input", history_messages_key="chat_history",
        output_messages_key="output",
    )


def action_prompt(action: str, text: str, language: str = "English") -> str:
    """Ask the agent for the selected action; Ask preserves the original input."""
    instructions = {
        "Fix Layout": "Fix keyboard layout",
        "Translate": f"Translate to {language}",
        "Grammar": "Check grammar",
        "Save Note": "Save note",
    }
    if action == "Ask":
        return text
    return f"{instructions[action]}: {text}"


def is_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    response_status = getattr(getattr(exc, "response", None), "status_code", None)
    message = f"{type(exc).__name__} {exc}".lower()
    return (
        status in (413, 429) or response_status in (413, 429)
        or "rate_limit" in message or "rate limit" in message
        or "ratelimit" in message or bool(re.search(r"\b(?:413|429)\b", message))
    )


def safe_error(exc: Exception) -> str:
    message = str(exc)
    for key in ("GROQ_API_KEY", "LANGSMITH_API_KEY"):
        secret = os.getenv(key)
        if secret:
            message = message.replace(secret, "[redacted]")
    return f"Error ({type(exc).__name__}): {message[:1500]}"


class ClipboardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events = queue.Queue()
        self.stopped = threading.Event()
        self.monitor = threading.Event()
        self.agent = None
        self.busy = False
        self.keyboard = None
        self.hotkey = None
        self.highlight_job = None
        self.copy_reset_job = None
        self._clipboard_lock = threading.Lock()
        self._self_copied_text = None
        root.title("Study & Clipboard Agent")
        root.geometry("400x500")
        root.minsize(360, 440)
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(root, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Text or question").pack(anchor="w")
        self.input = ScrolledText(frame, height=5, wrap="word", undo=True)
        self.input.pack(fill="x", pady=(4, 6))
        self.input_background = self.input.cget("background")

        language_row = ttk.Frame(frame)
        language_row.pack(fill="x", pady=(0, 6))
        ttk.Label(language_row, text="Translate to:").pack(side="left")
        self.language = tk.StringVar(value="English")
        ttk.Combobox(
            language_row, textvariable=self.language, state="readonly", width=12,
            values=("English", "Russian", "Kazakh"),
        ).pack(side="left", padx=6)

        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        self.buttons = []
        for index, action in enumerate((
            "Fix Layout", "Translate", "Grammar", "Save Note",
        )):
            button = ttk.Button(actions, text=action, command=lambda a=action: self.submit(a))
            button.grid(row=0, column=index, sticky="ew", padx=1, pady=2)
            self.buttons.append(button)
        for column in range(4):
            actions.columnconfigure(column, weight=1, uniform="actions")
        ask_button = ttk.Button(actions, text="Ask", command=lambda: self.submit("Ask"))
        ask_button.grid(row=1, column=0, columnspan=4, sticky="ew", padx=1, pady=(4, 2), ipady=3)
        self.buttons.append(ask_button)

        result_header = ttk.Frame(frame)
        result_header.pack(fill="x", pady=(8, 4))
        ttk.Label(result_header, text="Result").pack(side="left")
        self.copy_button = ttk.Button(result_header, text="📋 Copy", command=self.copy_result)
        self.copy_button.pack(side="left", padx=(8, 0))
        self.output = ScrolledText(frame, wrap="word", state="disabled", height=10)
        self.output.pack(fill="both", expand=True)
        bottom_bar = ttk.Frame(frame)
        bottom_bar.pack(fill="x", pady=(8, 0))
        self.toggle = ttk.Button(bottom_bar, text="Clipboard Monitor OFF", command=self.toggle_monitor)
        self.toggle.pack(side="left")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(bottom_bar, textvariable=self.status, relief="sunken", anchor="w").pack(
            side="left", fill="x", expand=True, padx=(6, 0),
        )

        root.after(100, self.drain_events)
        threading.Thread(target=self.watch_clipboard, daemon=True).start()
        self.register_hotkey()

    def show_output(self, text: str):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def copy_result(self):
        result_text = self.output.get("1.0", "end-1c")
        if not result_text.strip():
            return
        try:
            if pyperclip is None:
                raise RuntimeError("Copy requires pyperclip. Install it with: pip install pyperclip")
            with self._clipboard_lock:
                previous = self._self_copied_text
                self._self_copied_text = result_text
                try:
                    pyperclip.copy(result_text)
                except Exception:
                    self._self_copied_text = previous
                    raise
        except Exception as exc:
            self.status.set(safe_error(exc))
            return
        self.copy_button.configure(text="✓ Copied")
        if self.copy_reset_job is not None:
            self.root.after_cancel(self.copy_reset_job)
        self.copy_reset_job = self.root.after(1500, self.reset_copy_button)

    def reset_copy_button(self):
        self.copy_button.configure(text="📋 Copy")
        self.copy_reset_job = None

    def refresh_status(self):
        self.status.set("Processing..." if self.busy else (
            "Clipboard mode: ON" if self.monitor.is_set() else "Ready"
        ))

    def submit(self, action: str):
        if self.busy:
            return
        input_text = self.input.get("1.0", "end-1c").strip()
        if not input_text:
            self.show_output(
                "Nothing to save — input is empty."
                if action == "Save Note" else "Please enter some text first."
            )
            return
        text = input_text[:1500]
        prompt = action_prompt(action, text, self.language.get())
        self.busy = True
        for button in self.buttons:
            button.configure(state="disabled")
        self.refresh_status()
        threading.Thread(target=self.run_agent, args=(prompt,), daemon=True).start()

    def run_agent(self, prompt: str):
        try:
            if self.agent is None:
                self.agent = build_agent()
            result = self.agent.invoke(
                {"input": prompt}, config={"configurable": {"session_id": "desktop"}},
            )
            self.events.put(("result", str(result["output"])))
        except Exception as exc:
            message = "Text too long, please shorten it" if is_limit_error(exc) else safe_error(exc)
            self.events.put(("result", message))

    def toggle_monitor(self):
        if self.monitor.is_set():
            self.monitor.clear()
            self.toggle.configure(text="Clipboard Monitor OFF")
        elif pyperclip is None:
            self.show_output("Clipboard monitoring requires pyperclip. Install it with: pip install pyperclip")
        else:
            self.monitor.set()
            self.toggle.configure(text="Clipboard Monitor ON")
        self.refresh_status()

    def watch_clipboard(self):
        last_text = None
        while not self.stopped.wait(2):
            if not self.monitor.is_set():
                last_text = None
                continue
            try:
                with self._clipboard_lock:
                    text = pyperclip.paste()
                    # Keep this marker until a poll observes the write. A flag
                    # cleared immediately after copy would miss the 2s poll.
                    if self._self_copied_text is not None:
                        copied_text = self._self_copied_text
                        self._self_copied_text = None
                        if text == copied_text:
                            last_text = text
                            continue
                    if text != last_text:
                        if last_text is not None and isinstance(text, str) and text.strip():
                            self.events.put(("clipboard", text))
                        last_text = text
            except Exception as exc:
                self.monitor.clear()
                self.events.put(("clipboard_error", safe_error(exc)))

    def register_hotkey(self):
        try:
            import keyboard
            self.keyboard = keyboard
            self.hotkey = keyboard.add_hotkey(
                "ctrl+shift+a", lambda: self.events.put(("hotkey", None)),
            )
        except Exception:
            pass  # Optional: keyboard may be absent or global hooks unavailable.

    def paste_input(self, text: str):
        self.input.delete("1.0", "end")
        self.input.insert("1.0", text)
        self.input.configure(background="#fff5cc")
        if self.highlight_job is not None:
            self.root.after_cancel(self.highlight_job)
        self.highlight_job = self.root.after(900, self.clear_highlight)

    def clear_highlight(self):
        self.input.configure(background=self.input_background)
        self.highlight_job = None

    def drain_events(self):
        # Tk widgets are accessed only on this main-thread event loop.
        for _ in range(100):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "result":
                self.show_output(value)
                self.busy = False
                for button in self.buttons:
                    button.configure(state="normal")
                self.refresh_status()
            elif kind == "clipboard":
                if self.monitor.is_set():
                    self.paste_input(value)
            elif kind == "clipboard_error":
                self.toggle.configure(text="Clipboard Monitor OFF")
                self.show_output(value)
                self.refresh_status()
            elif kind == "hotkey":
                self.root.deiconify()
                self.root.lift()
                self.root.focus_force()
                try:
                    # Main-thread Tk fallback keeps the hotkey useful without pyperclip.
                    text = self.root.clipboard_get()
                    self.paste_input(text)
                    self.input.focus_set()
                except tk.TclError:
                    self.show_output("The clipboard does not contain text.")
        if not self.stopped.is_set():
            self.root.after(100, self.drain_events)

    def close(self):
        self.stopped.set()
        self.monitor.clear()
        if self.keyboard is not None and self.hotkey is not None:
            try:
                self.keyboard.remove_hotkey(self.hotkey)
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    ClipboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
