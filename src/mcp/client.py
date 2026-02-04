import os
from dataclasses import dataclass
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict
    server_name: str


class MCPClient:
    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tools: list[MCPTool] = []
        self._exit_stack = AsyncExitStack()

    async def connect(self, server_name: str, command: str, args: list[str] | None = None):
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env={**os.environ},
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = ClientSession(read, write)
        await self._exit_stack.enter_async_context(session)
        await session.initialize()

        self.sessions[server_name] = session
        await self._load_tools(server_name, session)

    async def _load_tools(self, server_name: str, session: ClientSession):
        result = await session.list_tools()
        for tool in result.tools:
            self.tools.append(
                MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    server_name=server_name,
                )
            )

    def get_openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        for tool in self.tools:
            if tool.name == tool_name:
                session = self.sessions.get(tool.server_name)
                if session:
                    try:
                        result = await session.call_tool(tool_name, arguments)
                        return "\n".join(
                            content.text for content in result.content if hasattr(content, "text")
                        )
                    except Exception as e:
                        return f"Tool error: {e}"
        return f"Tool '{tool_name}' not found"

    async def close(self):
        await self._exit_stack.aclose()
        self.sessions.clear()
        self.tools.clear()
