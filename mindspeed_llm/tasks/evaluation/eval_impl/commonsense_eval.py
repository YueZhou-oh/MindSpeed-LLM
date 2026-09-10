"""Native DatasetEval adapter for the six harness commonsense tasks."""

import logging
import time

import pandas as pd
from torch import distributed as dist

from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_utils.commonsense_data import (
    TASKS,
    canonical_task,
    format_example,
    load_documents,
    predictions,
    encode_pair,
)

logger = logging.getLogger(__name__)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class CommonsenseEval(DatasetEval):
    def __init__(self, test_dir, eval_args, task):
        self.test_dir = test_dir
        self.args = eval_args
        self.task = canonical_task(task)

    def eval(self, chat):
        rank = dist.get_rank() if dist.is_initialized() else 0
        is_main = rank == 0

        data = load_documents(self.test_dir, self.task)
        limit = getattr(self.args, "max_eval_samples", None)
        count = len(data) if limit is None else min(limit, len(data))

        if count <= 0:
            raise ValueError("Evaluation requires at least one example")

        batch_size = self.args.evaluation_batch_size
        if batch_size <= 0:
            raise ValueError("evaluation-batch-size must be positive")

        answers = {}
        correct = 0
        normalized = 0
        completed = 0

        if is_main:
            logger.info(
                "[%s] Starting evaluation: samples=%d/%d, "
                "evaluation_batch_size=%d",
                self.task, count, len(data), batch_size,
            )
            logger.info(
                "[%s] A/B/C/... represent candidate indices. "
                "acc uses summed log-likelihood; "
                "acc_norm uses character-normalized log-likelihood.",
                self.task,
            )

        start_time = time.perf_counter()

        # All TP ranks must execute the same model calls.
        # DP replicas evaluate identical data; do not sum their counts.
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch_start = time.perf_counter()

            examples = [
                format_example(self.task, data[index])
                for index in range(start, stop)
            ]

            requests = []

            for context, choices, _ in examples:
                if self.task == "winogrande":
                    # WinoGrande: different contexts, identical suffix.
                    if len(context) != len(choices):
                        raise ValueError(
                            "WinoGrande context/continuation count mismatch"
                        )

                    requests.extend(
                        (candidate_context, " " + continuation)
                        for candidate_context, continuation
                        in zip(context, choices)
                    )
                else:
                    # Other tasks: identical context, different answers.
                    requests.extend(
                        (context, " " + choice)
                        for choice in choices
                    )

            scores = chat.loglikelihood(requests)

            if len(scores) != len(requests):
                raise ValueError(
                    f"Expected {len(requests)} scores, got {len(scores)}"
                )

            batch_seconds = time.perf_counter() - batch_start
            completed = stop
            elapsed = time.perf_counter() - start_time
            seconds_per_sample = elapsed / completed
            eta = seconds_per_sample * (count - completed)

            offset = 0

            for index, (_, choices, gold) in enumerate(examples, start):
                values = scores[offset:offset + len(choices)]
                offset += len(choices)

                pred, pred_norm = predictions(values, choices)
                correct += int(pred == gold)
                normalized += int(pred_norm == gold)

                if is_main:
                    answers[str(index)] = {
                        "gold": gold,
                        "prediction": pred,
                        "prediction_norm": pred_norm,
                        "loglikelihood": values,
                    }

                    labels = [
                        chr(ord("A") + i) for i in range(len(choices))
                    ]
                    score_text = ", ".join(
                        f"{labels[i]}={value:.4f}"
                        for i, value in enumerate(values)
                    )
                    norm_text = ", ".join(
                        f"{labels[i]}={values[i] / len(choice):.4f}"
                        for i, choice in enumerate(choices)
                    )

                    logger.info(
                        "[%s] sample=%d/%d | gold=%s | "
                        "pred=%s (%s) | pred_norm=%s (%s) | "
                        "acc=%.4f | acc_norm=%.4f | "
                        "batch_s/sample=%.3f | avg_s/sample=%.3f | "
                        "elapsed=%s | ETA=%s | "
                        "loglikelihood=[%s] | normalized_scores=[%s]",
                        self.task,
                        index + 1,
                        count,
                        labels[gold],
                        labels[pred],
                        "correct" if pred == gold else "wrong",
                        labels[pred_norm],
                        "correct" if pred_norm == gold else "wrong",
                        correct / (index + 1),
                        normalized / (index + 1),
                        batch_seconds / len(examples),
                        seconds_per_sample,
                        format_duration(elapsed),
                        format_duration(eta),
                        score_text,
                        norm_text,
                    )

        columns = ["subject", "question_n", "acc", "acc_norm"]

        if not is_main:
            return {}, pd.DataFrame(columns=columns)

        elapsed = time.perf_counter() - start_time
        rows = [
            [self.task, count, correct / count, normalized / count],
            ["total", count, correct / count, normalized / count],
        ]
        score_df = pd.DataFrame(rows, columns=columns)

        logger.info(
            "[%s] Finished | samples=%d | "
            "correct=%d | correct_norm=%d | "
            "acc=%.6f | acc_norm=%.6f | "
            "elapsed=%s | avg_s/sample=%.3f",
            self.task,
            count,
            correct,
            normalized,
            correct / count,
            normalized / count,
            format_duration(elapsed),
            elapsed / count,
        )
        logger.info("\n%s", score_df.to_string(index=False))

        return {self.task: answers}, score_df

    def top_k_eval(self):
        raise NotImplementedError(
            "Use eval() for multiple-choice likelihood scoring"
        )