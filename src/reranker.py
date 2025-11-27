"""Reranking module using Korean cross-encoder for better relevance scoring"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config.config as cfg


class Reranker:
    """Cross-encoder based reranker for improving search results"""

    def __init__(
        self,
        model_name: str = cfg.RERANKER_MODEL,
        device: str = cfg.RERANKER_DEVICE,
        max_length: int = cfg.RERANKER_MAX_LENGTH,
        batch_size: int = cfg.RERANKER_BATCH_SIZE,
    ):
        """Initialize Korean cross-encoder reranker"""

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Loading reranker model: {model_name}")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
        )
        self.model.to(self.device)
        self.model.eval()

        if self.device == 'cuda':
            logger.info("Using GPU for reranking")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return (None = all)

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []

        logger.debug(f"Reranking {len(documents)} documents")

        scores = self._score_documents(query, [doc['content'] for doc in documents])

        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
            doc['original_score'] = doc.get('score', 0)

        # Sort by reranking score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        # Return top-k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.debug(f"Reranking complete. Top score: {reranked[0]['rerank_score']:.4f}")

        return reranked

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Score query-text pairs

        Args:
            query: Search query
            texts: List of texts

        Returns:
            List of relevance scores
        """
        return self._score_documents(query, texts)

    def _score_documents(self, query: str, texts: List[str]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).detach().cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
            scores.extend(batch_scores)
        return scores


class _RerankerTrainingDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
        default_weight: float = 1.0,
        hard_negative_weight: float = 1.0
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_weight = default_weight
        self.hard_negative_weight = hard_negative_weight

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        encoded = self.tokenizer(
            sample['query'],
            sample['content'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item['labels'] = torch.tensor(float(sample['label']), dtype=torch.float)
        weight = sample.get('weight')
        if weight is None:
            weight = self.hard_negative_weight if sample.get('hard_negative') else self.default_weight
        item['sample_weight'] = torch.tensor(float(weight), dtype=torch.float)
        return item


class _WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        weights = inputs.pop('sample_weight', None)
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss_fct = torch.nn.MSELoss(reduction='none')
        loss = loss_fct(logits, labels)
        if weights is not None:
            loss = loss * weights
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


# Singleton instance
_reranker_instance = None

def get_reranker() -> Reranker:
    """Get singleton reranker instance"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


def _load_baseline(baseline_path: Path) -> Dict[int, Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    with open(baseline_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line.strip())
            entries[row['eval_id']] = {
                'query': row.get('standalone_query', '').strip(),
                'positives': set(row.get('topk', []))
            }
    return entries


def _prepare_dataset_cli(args):
    from src.retriever import ElasticsearchRetriever

    baseline = _load_baseline(Path(args.baseline))
    retriever = ElasticsearchRetriever()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out:
        for eval_id in sorted(baseline.keys()):
            query = baseline[eval_id]['query']
            positives = baseline[eval_id]['positives']

            if args.skip_empty and (not query or not positives):
                continue
            if not query:
                continue

            docs = retriever.hybrid_retrieve(query, size=args.topk)
            for rank, doc in enumerate(docs, start=1):
                label = 1 if doc['docid'] in positives else 0
                is_hard_negative = (
                    label == 0 and args.hard_negative_rank > 0 and rank <= args.hard_negative_rank
                )
                record = {
                    'eval_id': eval_id,
                    'query': query,
                    'docid': doc['docid'],
                    'content': doc['content'],
                    'label': label,
                    'rank': rank,
                    'sparse_score': doc.get('score', 0),
                    'hard_negative': is_hard_negative
                }
                out.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Dataset saved to {output_path}")


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def _train_cli(args):
    data_path = Path(args.data)
    samples = _load_samples(data_path)
    if not samples:
        raise RuntimeError(f"No samples found in {data_path}. Run the dataset preparation step first.")

    random.Random(42).shuffle(samples)
    split_idx = max(1, int(len(samples) * 0.1))
    eval_samples = samples[:split_idx]
    train_samples = samples[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        problem_type='regression'
    )

    train_dataset = _RerankerTrainingDataset(
        train_samples,
        tokenizer,
        args.max_length,
        default_weight=args.default_weight,
        hard_negative_weight=args.hard_negative_weight
    )
    eval_dataset = _RerankerTrainingDataset(
        eval_samples,
        tokenizer,
        args.max_length,
        default_weight=1.0,
        hard_negative_weight=1.0
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        logging_steps=50,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.squeeze()
        mse = float(np.mean((preds - labels) ** 2))
        return {'mse': mse}

    trainer = _WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info(f"Fine-tuned model saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Korean cross-encoder reranker utilities")
    subparsers = parser.add_subparsers(dest='command')

    prep = subparsers.add_parser('prepare', help='Build training dataset from baseline positives')
    prep.add_argument('--baseline', default=str(cfg.BASELINE_SAMPLE_PATH))
    prep.add_argument('--output', default=str(cfg.DATA_DIR / 'reranker_training.jsonl'))
    prep.add_argument('--topk', type=int, default=20)
    prep.add_argument('--skip-empty', action='store_true')
    prep.add_argument('--hard-negative-rank', type=int, default=5,
                      help='Ranks <= value treated as hard negatives when not in baseline positives')

    train = subparsers.add_parser('train', help='Fine-tune reranker on prepared dataset')
    train.add_argument('--data', default=str(cfg.DATA_DIR / 'reranker_training.jsonl'))
    train.add_argument('--output', default=str(cfg.BASE_DIR / 'models' / 'reranker'))
    train.add_argument('--model', default=cfg.RERANKER_MODEL)
    train.add_argument('--epochs', type=int, default=2)
    train.add_argument('--batch-size', type=int, default=8)
    train.add_argument('--lr', type=float, default=2e-5)
    train.add_argument('--max-length', type=int, default=cfg.RERANKER_MAX_LENGTH)
    train.add_argument('--hard-negative-weight', type=float, default=1.0,
                       help='Extra loss weight for hard negatives')
    train.add_argument('--default-weight', type=float, default=1.0,
                       help='Base sample weight for other examples')

    parsed = parser.parse_args()
    if parsed.command == 'prepare':
        _prepare_dataset_cli(parsed)
    elif parsed.command == 'train':
        _train_cli(parsed)
    else:
        parser.print_help()
