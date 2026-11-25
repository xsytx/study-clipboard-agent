"""Regression checks for the notebook's LLM content-block/history boundary."""

import asyncio
import json
import os
import unittest
from pathlib import Path

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


class AgentHistoryTests(unittest.TestCase):
    def setUp(self):
        self.history = InMemoryChatMessageHistory()
        self.steps = []
        self.calls = 0

        def executor(values):
            # Formatting the next prompt reproduces the original coercion failure
            # if raw content-block dictionaries have leaked into history.
            ChatPromptTemplate.from_messages([
                MessagesPlaceholder("chat_history"), ("human", "{input}"),
            ]).invoke(values)
            self.calls += 1
            if self.calls == 2:
                self.assertEqual(len(values["chat_history"]), 2)
                self.assertEqual(values["chat_history"][1].content, "Explanation")
            return {
                "output": [
                    {"type": "text", "text": "", "extras": {"signature": "test-signature"}, "index": 0},
                    {"type": "text", "text": "Explanation" if self.calls == 1 else "Flashcards created"},
                ],
                "intermediate_steps": self.steps,
            }

        notebook = json.loads(Path(__file__).with_name("agent.ipynb").read_text(encoding="utf-8"))
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "def normalize_agent_output(" in "".join(cell["source"]))
        # Execute the actual notebook's helper and history wiring.
        self.namespace = {
            "AIMessage": AIMessage, "BaseMessage": BaseMessage,
            "StrOutputParser": StrOutputParser, "RunnableLambda": RunnableLambda,
            "ChatMessageHistory": InMemoryChatMessageHistory,
            "BaseChatMessageHistory": InMemoryChatMessageHistory,
            "RunnableWithMessageHistory": RunnableWithMessageHistory,
            "agent_executor": RunnableLambda(executor),
        }
        exec(compile(source, "<notebook history cell>", "exec"), self.namespace)
        self.chain = self.namespace["agent_with_history"]
        self.config = {"configurable": {"session_id": "regression"}}

    def check_result(self, result):
        self.assertIsInstance(result["output"], str)
        self.assertIs(result["intermediate_steps"], self.steps)

    def check_history(self):
        messages = self.namespace["store"]["regression"].messages
        self.assertEqual(len(messages), 4)
        self.assertTrue(all(isinstance(message, BaseMessage) for message in messages))
        self.assertEqual(messages[1].content, "Explanation")
        self.assertEqual(messages[3].content, "Flashcards created")

    def test_signed_content_blocks_across_two_turns(self):
        for prompt in ("Explain backpropagation", "make flashcards from that"):
            self.check_result(self.chain.invoke({"input": prompt}, self.config))
        self.check_history()

    def test_async_signed_content_blocks_across_two_turns(self):
        async def run():
            for prompt in ("Explain backpropagation", "make flashcards from that"):
                self.check_result(await self.chain.ainvoke({"input": prompt}, self.config))
        asyncio.run(run())
        self.check_history()

    def test_string_and_message_outputs(self):
        normalize = self.namespace["normalize_agent_output"]
        original = {"output": "Plain answer", "intermediate_steps": self.steps}
        self.assertIs(normalize(original), original)
        result = normalize({"output": AIMessage(content=[{"type": "text", "text": "Answer"}])})
        self.assertEqual(result["output"], "Answer")


if __name__ == "__main__":
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    unittest.main()
