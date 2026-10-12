# -*- coding: utf-8 -*-
"""검색·답변 지표.

작은 셋(40문항)에서는 전체 평균이 서사를 만들지 못한다. 유형별로 쪼개야
"조항번호형이 Hybrid 로 얼마나 올랐는가" 같은 말을 할 수 있다.
"""
from __future__ import annotations

from collections import defaultdict


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """gold 청크가 상위 k 안에 하나라도 있으면 1. 근거는 여러 개일 수 있다."""
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def mrr(retrieved: list[str], gold: list[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in set(gold):
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not k:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / k


def aggregate(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """행 하나 = 문항 하나. 전체 평균과 type 별 분해표를 같이 만든다.

    반환 형태 (results/*.json 의 overall · by_type 과 같다)
        {"overall": 블록, "by_type": {유형: 블록, ...}}
    블록 하나에 들어가는 것
        recall@k (ks 마다) · mrr · n                      — 전체 평균. 소수 4자리
        answer_rate                                       — answered 키가 있는 행이 하나라도 있을 때만.
                                                            거절하지 않고 답한 비율
        false_refusal_rate                                — answerable 인 행이 있을 때만. 답할 근거가
                                                            있는데(answerable=True) 거절한 비율
        abstain_accuracy                                  — answerable=False 인 행이 있을 때만. 거절했거나
                                                            grade 가 "insufficient"(한계를 밝힌 답)인 비율
        latency_ms_avg · latency_ms_p95                   — latency_ms 가 있는 행이 있을 때만.
                                                            p95 는 정렬한 뒤 int(n × 0.95) − 1 번째(0 미만이면 0)

    [6회차 실습 1] 이 함수를 채우세요. 테스트는 tests/test_metrics.py 입니다.
    왜 유형별인가: 40문항짜리 셋에서 전체 평균 하나로는 아무 이야기도 못 만든다. "조항번호형이
    Hybrid 로 얼마나 올랐는가"를 말하려면 유형별로 쪼개야 한다.
    왜 분모를 가르나: 오거절률(답할 수 있는데 거절)과 미답변 정확도(거절해야 할 것을 거절)는
    분모가 다르다. 둘을 한 숫자로 합치면 "전부 거절"이 좋은 점수를 받는다.
    """
    def block(subset: list[dict]) -> dict:
        n = len(subset)
        out: dict = {"n": n}
        if n == 0:
            return out

        def mean(vals: list[float], nd: int = 4) -> float:
            return round(sum(vals) / len(vals), nd)

        # 검색 지표. 전체 평균. 키가 없는 행은 그 지표의 분모에서 뺀다.
        for k in ks:
            key = f"recall@{k}"
            vals = [float(r[key]) for r in subset if key in r]
            if vals:
                out[key] = mean(vals)
        vals = [float(r["mrr"]) for r in subset if "mrr" in r]
        if vals:
            out["mrr"] = mean(vals)

        # 답변 지표. baseline · hybrid 결과에는 answered 가 없으니 자리를 만들지 않는다.
        with_ans = [r for r in subset if "answered" in r]
        if with_ans:
            out["answer_rate"] = mean([1.0 if r["answered"] else 0.0 for r in with_ans])
            # 오거절률: 답할 근거가 있는데(answerable) 거절한 비율. 분모는 answerable 행만.
            answerable = [r for r in with_ans if r.get("answerable", True)]
            if answerable:
                out["false_refusal_rate"] = mean([0.0 if r["answered"] else 1.0 for r in answerable])
            # 미답변 정확도: 거절해야 할 문항 중 거절했거나 한계를 밝힌(insufficient) 비율. 분모는 그 행만.
            unanswerable = [r for r in with_ans if not r.get("answerable", True)]
            if unanswerable:
                out["abstain_accuracy"] = mean([
                    1.0 if (not r["answered"] or r.get("grade") == "insufficient") else 0.0
                    for r in unanswerable])

        # 지연. p95 는 정렬한 뒤 int(n × 0.95) − 1 번째(0 미만이면 0).
        lat = sorted(float(r["latency_ms"]) for r in subset if r.get("latency_ms") is not None)
        if lat:
            out["latency_ms_avg"] = round(sum(lat) / len(lat), 1)
            out["latency_ms_p95"] = round(lat[max(int(len(lat) * 0.95) - 1, 0)], 1)
        return out

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[str(r.get("type", "기타"))].append(r)
    return {"overall": block(rows),
            "by_type": {t: block(rs) for t, rs in sorted(by_type.items())}}


def compare(tables: dict[str, dict], metric: str = "recall@5") -> list[dict]:
    """3단계 비교표. 파이프라인별 유형 분해를 한 표로 세운다."""
    types = sorted({t for tb in tables.values() for t in tb["by_type"]})
    rows = []
    for t in types:
        row = {"type": t}
        for name, tb in tables.items():
            row[name] = tb["by_type"].get(t, {}).get(metric)
        rows.append(row)
    row = {"type": "전체"}
    for name, tb in tables.items():
        row[name] = tb["overall"].get(metric)
    rows.append(row)
    return rows
