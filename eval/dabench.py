import json
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/InfiAgent/InfiAgent/main/examples/DA-Agent/data"
DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Item:
    id: int
    question: str
    constraints: str
    answer_format: str
    file_name: str
    level: str
    labels: list[tuple[str, str]]

    @property
    def csv_path(self) -> str:
        return str(DATA_DIR / "tables" / self.file_name)

    def task_text(self) -> str:
        return (
            f"{self.question}\n\n"
            f"Constraints: {self.constraints}\n\n"
            f"Answer format: {self.answer_format}\n\n"
            "After your analysis, end your final answer with the results strictly "
            "in the specified @name[value] format."
        )


def _download(remote_name: str, local_path: Path) -> None:
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE}/{urllib.parse.quote(remote_name)}"
    print(f"Downloading {remote_name}...")
    urllib.request.urlretrieve(url, local_path)


def load_items(n: int | None = None, seed: int = 0) -> list[Item]:
    """
    Load DABench dev items with labels, downloading data on first use.

    When n is given, a seeded sample stratified by difficulty level is returned
    so repeated runs use the same subset. Only the sampled items' CSV tables
    are downloaded.
    """
    questions_path = DATA_DIR / "da-dev-questions.jsonl"
    labels_path = DATA_DIR / "da-dev-labels.jsonl"
    _download("da-dev-questions.jsonl", questions_path)
    _download("da-dev-labels.jsonl", labels_path)

    labels_by_id = {}
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            labels_by_id[record["id"]] = [tuple(pair) for pair in record["common_answers"]]

    items = []
    with questions_path.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["id"] not in labels_by_id:
                continue
            items.append(Item(
                id=q["id"],
                question=q["question"],
                constraints=q.get("constraints", ""),
                answer_format=q.get("format", ""),
                file_name=q["file_name"],
                level=q.get("level", "unknown"),
                labels=labels_by_id[q["id"]],
            ))

    if n is not None and n < len(items):
        items = _stratified_sample(items, n, seed)

    for item in items:
        _download(f"da-dev-tables/{item.file_name}", DATA_DIR / "tables" / item.file_name)
    return items


def _stratified_sample(items: list[Item], n: int, seed: int) -> list[Item]:
    rng = random.Random(seed)
    by_level: dict[str, list[Item]] = {}
    for item in items:
        by_level.setdefault(item.level, []).append(item)

    sampled: list[Item] = []
    levels = sorted(by_level)
    for i, level in enumerate(levels):
        # Distribute n across levels proportionally, giving remainders to later levels.
        quota = round(n * len(by_level[level]) / len(items))
        if i == len(levels) - 1:
            quota = n - len(sampled)
        sampled.extend(rng.sample(by_level[level], min(quota, len(by_level[level]))))
    return sorted(sampled[:n], key=lambda item: item.id)
