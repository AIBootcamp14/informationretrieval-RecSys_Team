"""
Main script for IR system - Index documents and run evaluation
"""
import json
import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config.config as cfg
from src.rag_pipeline import RAGPipeline
from chunker import load_documents, chunk_documents


def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=cfg.LOG_LEVEL
    )

    # File handler
    logger.add(
        cfg.LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )

    logger.info("Logging initialized")


def load_docs_for_index(filepath: Path) -> list:
    """Load documents from JSONL file"""
    logger.info(f"Loading documents from {filepath}")
    documents = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def load_eval_data(filepath: Path) -> list:
    """Load evaluation data from JSONL file"""
    logger.info(f"Loading evaluation data from {filepath}")
    eval_data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            eval_data.append(item)

    logger.info(f"Loaded {len(eval_data)} evaluation items")
    return eval_data


def index_documents(pipeline: RAGPipeline, documents_file: Path):
    """Index documents and chunks locally"""
    logger.info("=" * 60)
    logger.info("STEP 1: Indexing Documents")
    logger.info("=" * 60)

    # Load and chunk documents
    documents = load_documents(documents_file)
    chunks = chunk_documents(documents)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # Index chunks (BM25 + BGE-M3)
    pipeline.index_documents(documents, chunks)

    logger.info("Document indexing complete!")


def run_evaluation(pipeline: RAGPipeline, eval_file: Path, output_file: Path):
    """Run evaluation on eval dataset"""
    logger.info("=" * 60)
    logger.info("STEP 2: Running Evaluation")
    logger.info("=" * 60)

    eval_data = load_eval_data(eval_file)

    results = []
    with open(output_file, 'w', encoding='utf-8') as of:
        for item in tqdm(eval_data, desc="Processing eval data"):
            eval_id = item['eval_id']
            messages = item['msg']

            logger.info(f"\n{'='*60}")
            logger.info(f"Eval ID: {eval_id}")
            logger.info(f"Messages: {messages}")

            # Get answer
            response = pipeline.answer_question(messages, eval_id=eval_id)

            # Prepare output
            output = {
                "eval_id": eval_id,
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }

            # Write to file
            of.write(json.dumps(output, ensure_ascii=False) + '\n')
            results.append(output)

            logger.info(f"Standalone Query: {response['standalone_query']}")
            logger.info(f"Top-K: {response['topk']}")
            logger.info(f"Answer: {response['answer']}")

    logger.info(f"\nEvaluation complete! Results saved to {output_file}")
    return results


def main():
    """Main execution function"""
    setup_logging()

    logger.info("=" * 60)
    logger.info("Enhanced IR System with RAG")
    logger.info("=" * 60)

    # File paths
    documents_file = cfg.DATA_DIR / "documents.jsonl"
    eval_file = cfg.DATA_DIR / "eval.jsonl"

    # Create submission folder if it doesn't exist
    submission_dir = cfg.BASE_DIR / "submission"
    submission_dir.mkdir(exist_ok=True)
    output_file = submission_dir / "submission.csv"

    # Check if files exist
    if not documents_file.exists():
        logger.error(f"Documents file not found: {documents_file}")
        logger.error("Please copy documents.jsonl to IR/data/ folder")
        return

    if not eval_file.exists():
        logger.error(f"Eval file not found: {eval_file}")
        logger.error("Please copy eval.jsonl to IR/data/ folder")
        return

    # Initialize pipeline
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # Step 1: Index documents (comment out if already indexed)
    index_documents(pipeline, documents_file)

    # Step 2: Run evaluation
    results = run_evaluation(pipeline, eval_file, output_file)

    logger.info("\n" + "=" * 60)
    logger.info("All tasks completed successfully!")
    logger.info(f"Total evaluations: {len(results)}")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
