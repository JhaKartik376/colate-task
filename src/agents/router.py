from openai import AsyncOpenAI

from src.config import get_config
from .specialized import QAAgent, SummaryAgent, ComparisonAgent


class AgentRouter:
    AGENT_MAP = {
        "qa": QAAgent,
        "summary": SummaryAgent,
        "comparison": ComparisonAgent,
    }

    def __init__(self, mcp_client=None):
        config = get_config()
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm_model
        self.mcp_client = mcp_client
        self.agents = {
            name: agent_class(self.mcp_client)
            for name, agent_class in self.AGENT_MAP.items()
        }

    async def route_async(self, query: str) -> str:
        agent_type = await self._classify_intent(query)
        agent = self.agents.get(agent_type, self.agents["qa"])
        return await agent.process_async(query)

    async def _classify_intent(self, query: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Classify the user query into one of these categories:
- qa: Direct questions seeking specific answers
- summary: Requests for overviews, summaries, or explanations of topics
- comparison: Requests to compare, contrast, or analyze differences

Respond with only the category name.""",
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=10,
        )

        intent = response.choices[0].message.content.strip().lower()
        return intent if intent in self.AGENT_MAP else "qa"
