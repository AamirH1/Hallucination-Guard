"""Actually executes the baseline-vs-framework benchmark and writes evaluation/run_report.md."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.evaluation.benchmark import run_benchmark
from app.evaluation.report import render_markdown
from app.providers.factory import get_embedding_provider, get_llm_provider
from app.retrieval.corpus import index_corpus
from app.retrieval.retrieval_factory import get_bm25_index
from app.retrieval.vector_store_factory import get_vector_store


async def main() -> None:
    settings = get_settings()
    embeddings = get_embedding_provider(settings)
    llm = get_llm_provider(settings)

    await index_corpus("data/sample_docs", get_vector_store(settings), get_bm25_index(), embeddings)

    print(f"Running benchmark with LLM_PROVIDER={settings.llm_provider}, EMBEDDING_PROVIDER={settings.embedding_provider}...")
    report = await run_benchmark(llm, embeddings)

    settings_desc = (
        f"llm_provider={settings.llm_provider}, embedding_provider={settings.embedding_provider}, "
        f"grounding_threshold={settings.grounding_threshold}, retrieval_threshold={settings.retrieval_threshold}"
    )
    markdown = render_markdown(report, settings_desc)

    out_path = Path("evaluation/run_report.md")
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {out_path} ({len(report.results)} case results)")
    print()
    print(markdown)


if __name__ == "__main__":
    asyncio.run(main())
