# Study & Clipboard Agent

A LangChain study assistant for selected or pasted clipboard text. It fixes mistyped keyboard layouts, translates between Kazakh/Russian/English, explains concepts, corrects English grammar, generates flashcards, keeps searchable notes, and requests live currency rates. Paste text into the notebook; this project does not read the operating system clipboard automatically.

## Tools

All nine functions in `tools.py` use `@tool`, typed arguments, clear docstrings, and `try/except`. Failures return readable error strings. Decorated tools also turn argument-validation failures into error observations.

1. `fix_layout(text)` — Automatically convert mistyped text between EN and RU physical key positions, preserving case.
2. `detect_language(text)` — Identify `kz`, `ru`, `en`, or `unknown` using Kazakh letters and Cyrillic/Latin counts.
3. `translate_text(text, target_language)` — Translate with Groq into `en`, `ru`, or `kz`.
4. `explain_selection(text)` — Explain a concept in simple terms, in at most 120 words.
5. `grammar_check(text)` — Return corrected English text and a short list of changes.
6. `make_flashcards(text)` — Generate Pydantic flashcards, export Anki CSV, and attempt AnkiConnect import.
7. `save_note(content, tags="")` — Save a timestamped note in SQLite and return its ID.
8. `search_notes(query)` — Search saved content and tags using parameterized SQL `LIKE`.
9. `convert_currency(amount, from_currency, to_currency)` — Request ExchangeRate API rates over HTTP (including KZT) and return a converted amount, unit rate, and rate date.

Layout detection uses the dominant alphabet; equal Latin/Cyrillic counts choose EN→RU. Text without letters is unchanged. Language detection is a heuristic: Kazakh without distinctive letters can appear Russian, equal counts return `unknown`, and empty text returns an error. SQLite's built-in `LIKE` does not provide full Unicode case folding.

## Files

```text
agent-project/
├── agent.ipynb
├── tools.py
├── models.py
├── config.py
├── db.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

Runtime creates `notes.db` and `exports/flashcards_<UTC timestamp>_<ID>.csv`; both are ignored by Git. Credentials belong only in the ignored `.env` file.

## Model and compatibility

The requested model is **Groq**, using `ChatGroq` from `langchain-groq` with model `openai/gpt-oss-120b`. The notebook explicitly initializes `get_llm()`.

The default is `openai/gpt-oss-120b`. For live use, confirm model availability for your Groq account and set `LLM_MODEL` in `.env` to an available model supporting tool calling and structured output if needed. Consult [Groq's available models](https://console.groq.com/docs/models).

LangChain 1.x exposes the requested `create_tool_calling_agent` and `AgentExecutor` through `langchain-classic`. See the [official factory reference](https://reference.langchain.com/python/langchain-classic/agents/tool_calling_agent/base/create_tool_calling_agent). This project intentionally retains that API and `RunnableWithMessageHistory`, as requested, even though newer LangChain releases recommend other agent and persistence APIs. Dependencies are pinned for the core integration; use Python 3.11+.

## Run

Clone the repository containing this project, then run these PowerShell commands from its root:

```powershell
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
cd agent-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="Study & Clipboard Agent"
LLM_MODEL=openai/gpt-oss-120b
RUN_LIVE_HTTP=true
```

Replace both key placeholders and confirm model availability before live calls. Obtain keys from [Groq Console](https://console.groq.com/keys) and [LangSmith](https://smith.langchain.com/). Configuration loads `.env` beside `config.py` without printing keys. Existing environment variables take precedence; restart the kernel after changing credentials or model settings.

Start Jupyter:

```powershell
python -m notebook agent.ipynb
```

Select the environment's Python kernel and run the notebook from top to bottom. Alternatively, open it in an editor with Jupyter support and select that environment. Cell 1 installs the same requirements using `%pip`.

The notebook contains eleven ordered code cells: dependency installation, imports, environment/tracing, database initialization, LLM initialization, tools, prompt, agent/executor, session memory, conversations, and the persistence proof.

Without both keys, live LLM calls are explicitly skipped. Local layout, language, and note tools still run. Direct currency checks run without keys when `RUN_LIVE_HTTP=true`; set it to `false` for an offline run. That flag only controls direct notebook checks; an agent calling `convert_currency` always invokes the real HTTP implementation.

## Example conversations and recorded outputs

The notebook includes these exact user prompts. The configured Groq model is `openai/gpt-oss-120b`. Notebook outputs have been cleared for publication; Groq live execution has not been reverified. The excerpts below describe the earlier offline behavior; rerun the notebook to record current live answers, tool calls, and exchange rates.

Example 1, copied from notebook output:

```text
=== Example 1: single turn ===
User: руддщ
Direct tool check (agent skipped): hello
```

With keys, the notebook sends `руддщ` to the agent, prints its `fix_layout` call, and displays the final response. Cell 6 also records the reverse conversion:

```text
Layout check: hello
Reverse layout check: привет
Language check: kz
```

Example 2, copied from the offline notebook output:

```text
=== Example 2: multi-turn memory ===
User: Explain backpropagation
Skipped live agent response: add API keys and select an available Groq model.
User: make flashcards from that
Skipped live agent response: add API keys and select an available Groq model.
```

On a live run, the first turn should call `explain_selection`; the second should call `make_flashcards` with the earlier explanation. Both use the same unique study session ID. Assertions verify that four user/assistant messages remain in history and that the expected tools were called. Flashcard contents depend on Groq and are not prewritten answers.

Example 3, copied from the offline notebook output:

```text
=== Example 3: real API, requested KZT pair ===
User: Convert 100 USD to KZT
Skipped live currency request: RUN_LIVE_HTTP=false.

=== Additional supported live pair ===
User: Convert 100 USD to EUR
Skipped live currency request: RUN_LIVE_HTTP=false.
```

The real HTTP implementation calls:

```text
https://open.er-api.com/v6/latest/USD
```

ExchangeRate API is free, requires no API key, and supports KZT, including USD to KZT and USD to EUR conversions. Its base endpoint is `https://open.er-api.com/v6/latest/{base}`. The result uses `rates[to_currency]` as the unit rate, multiplies it by the requested amount, and reports `time_last_update_utc`. Invalid currencies, API failures, and timeouts return error strings. Rates are the latest published rates rather than intraday market prices; see the [ExchangeRate API open-access documentation](https://www.exchangerate-api.com/docs/free).

Cell 11 saves a note, verifies it from a separate Python process, and searches it through the tool. A successful recorded line is:

```text
Persistence check: saved note reopened in a separate Python process.
```

The output also includes the actual generated note ID, UTC timestamp, and unique tag. Restart the kernel and search the printed tag to retrieve that note from `notes.db`.

## Flashcards and Anki

`Flashcard(front: str, back: str, tag: str)` and `FlashcardSet(cards: List[Flashcard])` are defined in `models.py`. `make_flashcards` uses `with_structured_output(FlashcardSet, method="json_schema")`, validates the result, and returns `List[Flashcard]` on success. Its return annotation is `Union[List[Flashcard], str]` so errors can be returned as strings, as required.

CSV is saved before attempting AnkiConnect. Files use UTF-8 with a BOM and proper CSV quoting, without a header row. Import into Anki as comma-separated text: map column 1 to **Front**, column 2 to **Back**, and column 3 to **Tags**. The CSV stores plain text; leave HTML interpretation disabled during import. Filenames and Anki status are printed in notebook output.

For the bonus import, run Anki with AnkiConnect listening on `http://localhost:8765`. The project creates the **Study & Clipboard Agent** deck and uses the **Basic** note type with **Front** and **Back** fields. A missing add-on, unavailable port, incompatible note type, or duplicate/import failure preserves the CSV and returns the cards. Network calls have timeouts. No Anki setup is necessary for CSV export.

## How the pieces work

1. `ChatPromptTemplate` combines the exact requested system prompt, the user's message, prior `chat_history`, and this turn's `agent_scratchpad`.
2. `create_tool_calling_agent` binds the nine tool schemas to Groq. The model chooses tool names and arguments.
3. `AgentExecutor` executes selected tools and feeds their observations back into the agent until a final response, bounded by iteration/time limits. `verbose=True` and intermediate steps show the actions.
4. LLM-backed text tools use LCEL: `prompt | llm | StrOutputParser()`. Flashcards use a Pydantic structured-output runnable instead of a string parser.
5. `RunnableWithMessageHistory` maps each session ID to `ChatMessageHistory`; `output_messages_key="output"` stores the final answer rather than intermediate executor fields. A `RunnableLambda` converts final Groq content blocks to text before the history wrapper sees them, preventing `MESSAGE_COERCION_FAILURE` while preserving internal tool-call signatures. Conversation memory lasts until kernel restart.
6. Notes use separate, persistent SQLite storage with parameterized inserts/searches and automatic UTC timestamps. The notebook verifies persistence by reopening SQLite in another process.
7. LangSmith traces the agent and its nested chains, including tool selection and outputs.

## LangSmith evidence

Tracing is enabled by both `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_TRACING=true`. Set `LANGSMITH_API_KEY` in `.env`; `LANGSMITH_PROJECT` defaults to **Study & Clipboard Agent**. Traces may include selected text and tool results.

After running with real credentials, open the project in [LangSmith](https://smith.langchain.com/) and replace these placeholders with your actual evidence:

- **[Screenshot placeholder: agent trace showing `fix_layout` and its observation]**
- **[Screenshot placeholder: explanation → flashcards follow-up, with session history]**
- **[Screenshot placeholder: `convert_currency` HTTP result used in the final answer]**
- **[Trace link placeholder: paste a shared LangSmith run link here]**

Do not commit `.env`, paste API keys into notebook cells, or include them in screenshots.

## Verification status

All project module imports and all imports in the notebook resolve in the configured Python 3.11 environment. All nine tools remain registered in `ALL_TOOLS`. PyPI confirms that `langchain-classic==1.1.0` and every other exact version in `requirements.txt` exist; the classic agent import is retained. The Notebook server is installed in the verification environment. The requested LangChain upgrade check left the installed versions unchanged, and the requirement pins match those installed versions. `pip check` reports no broken requirements.

Live ExchangeRate API checks successfully converted 100 USD to both KZT and EUR. Focused checks also covered input validation, currency-code normalization, zero and same-currency amounts, unsupported currencies, invalid rates, overflow, and request failures. The README and notebook no longer describe KZT as unsupported.

Before this provider update, all eleven notebook code cells executed successfully with the previous provider and saved credentials; Groq live execution has not been reverified. Example 2 called `explain_selection` followed by `make_flashcards`, returned no tool errors, and retained four valid chat-history messages. Notebook outputs have been cleared for publication; rerun locally to record fresh results. LangSmith tracing was enabled; delivery to the LangSmith UI was not independently inspected.

`python -m unittest test_agent_history -v` passes three regression checks covering the reported signature-bearing content blocks in synchronous and asynchronous two-turn history, plus plain-string and AIMessage outputs. Final responses are converted to text at the history boundary; internal tool-call signatures remain intact. Live example failures now raise instead of silently falling back, so notebook execution cannot conceal a failed agent turn.
