import json
import logging
from abc import ABC, abstractmethod

from openai import AsyncOpenAI
from rich.console import Console

from src.config import get_config

logger = logging.getLogger(__name__)
console = Console()


class BaseAgent(ABC):
    def __init__(self, mcp_client=None):
        config = get_config()
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm_model
        self.mcp_client = mcp_client
        self.max_iterations = 10

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    async def process_async(self, query: str) -> str:
        tools = self.mcp_client.get_openai_tools() if self.mcp_client else []

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        for iteration in range(self.max_iterations):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=0.3,
            )

            message = response.choices[0].message

            if message.tool_calls and self.mcp_client:
                messages.append(message)

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    console.print(f"  [dim]→ Calling tool:[/dim] [cyan]{tool_name}[/cyan]")
                    logger.info(f"Tool call: {tool_name}({arguments})")

                    result = await self.mcp_client.call_tool(tool_name, arguments)

                    console.print(f"  [dim]← Result:[/dim] [green]{len(result)} chars[/green]")
                    logger.debug(f"Tool result: {result[:200]}...")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            else:
                return message.content

        return "Max iterations reached"


class QAAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return """You are a precise Q&A agent. Use the available tools to search documents and answer questions.
Always search for relevant information before answering.
Cite sources in your response."""


class SummaryAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return """You are a summarization agent. Use the available tools to gather information from documents.
Search broadly to get comprehensive coverage of the topic.
Provide thorough but concise summaries."""


class ComparisonAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return """You are a comparison agent. Use the available tools to find information about different concepts.
Search for each concept separately to gather complete information.
Present comparisons in structured format (tables when helpful)."""
