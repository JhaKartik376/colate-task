from openai import OpenAI

from src.config import get_config
from src.vectordb import VectorStore


class RAGPipeline:
    def __init__(self, vector_store: VectorStore | None = None):
        config = get_config()
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.llm_model
        self.vector_store = vector_store or VectorStore()

    def query(self, question: str, top_k: int | None = None) -> str:
        search_results = self.vector_store.search(question, top_k)

        if not search_results:
            return "No relevant documents found to answer your question."

        context = self._build_context(search_results)
        return self._generate_response(question, context, search_results)

    def _build_context(self, search_results: list[dict]) -> str:
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result["metadata"]["source_file"]
            page = result["metadata"]["page_number"]
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{result['text']}")
        return "\n\n".join(context_parts)

    def _generate_response(
        self, question: str, context: str, search_results: list[dict]
    ) -> str:
        system_prompt = """You are a helpful research assistant. Answer questions based on the provided document context.
If the context doesn't contain enough information to answer the question, say so.
Always cite which source(s) you used in your answer."""

        user_prompt = f"""Context from documents:
{context}

Question: {question}

Provide a clear, accurate answer based on the context above."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        sources = []
        for result in search_results:
            source = f"{result['metadata']['source_file']} (Page {result['metadata']['page_number']})"
            if source not in sources:
                sources.append(source)

        return f"{answer}\n\n**Sources:** {', '.join(sources)}"
