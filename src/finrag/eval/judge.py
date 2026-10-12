# -*- coding: utf-8 -*-
"""금융 루브릭 LLM-as-Judge.

세 항목을 0·1·2 로 매긴다. 자유 서술로 점수를 매기게 하면 채점자가 흔들려서
같은 답변에 다른 점수가 나온다. 항목과 척도를 고정하고 근거를 요구한다.

이 judge 는 강사 수동 라벨 20개와 일치도를 재서 캘리브레이션한다(6회차).
일치도를 재지 않은 judge 점수는 숫자처럼 보이는 의견일 뿐이다.

6회차에서 쓰는 셋
  score_result  : 자기 agentic.json 의 20문항(JUDGE_SET)을 채점 → results/judge.json
  calibrate     : 강사 답변 20개(data/golden/judge_calibration.jsonl)를 같은 judge 로 채점해
                  강사 라벨과 일치도를 잰다. 자기 키로 돌려도 강사와 같은 답변을 채점하므로 비교가 된다.
  agreement     : 두 점수 열의 일치도(완전 일치 · ±1 · 평균 절대 오차). 6회차 실습 2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..llm import get_llm, is_fake

RUBRIC = """근거 일치(basis_match)
  0 gold 문서를 인용하지 않음
  1 gold 문서는 맞으나 조항·페이지가 다름
  2 gold 인용문이 포함된 청크를 인용

수치·단위(numeric_ok)
  0 값이 틀림
  1 값은 맞으나 단위·기준일·절사 규칙 표기 없음
  2 값·단위·규칙을 모두 표기
  (수치가 필요 없는 문항이면 2)

근거 밖 주장 없음(no_extra_claim)
  0 문서에 없는 사실을 진술
  1 애매한 일반론이 섞임
  2 인용 범위 안에서만 진술"""


class JudgeScore(BaseModel):
    basis_match: Literal[0, 1, 2] = Field(description="근거 일치")
    numeric_ok: Literal[0, 1, 2] = Field(description="수치·단위 정확성")
    no_extra_claim: Literal[0, 1, 2] = Field(description="근거 밖 주장 없음")
    reason: str = Field("", description="한 줄 이유")

    @property
    def total(self) -> int:
        return self.basis_match + self.numeric_ok + self.no_extra_claim


PROMPT = ChatPromptTemplate.from_messages([
    ("system", "너는 한국 금융문서 QA 채점자다. 아래 루브릭으로만 채점하라. "
               "친절함·문장력은 채점 대상이 아니다.\n\n" + RUBRIC),
    ("human", "질문: {question}\n\n정답 요지: {reference}\n\n"
              "정답 근거(gold): {gold}\n\n제출 답변: {answer}\n\n인용한 청크: {cited}"),
])


def score_one(question: str, answer: str, reference: str, gold: list[str],
              cited: list[str]) -> JudgeScore:
    llm = get_llm("judge", size="main")
    if is_fake(llm):
        # 키 없이도 형태는 유지한다. 근거 일치만 기계적으로 매긴다(모델 판단 아님).
        hit = 2 if set(cited) & set(gold) else 0
        return JudgeScore(basis_match=hit, numeric_ok=0, no_extra_claim=0,
                          reason="LLM 키가 없어 근거 일치만 기계 채점")
    chain = PROMPT | llm.with_structured_output(JudgeScore, method="json_schema")
    out = chain.invoke({"question": question, "answer": answer, "reference": reference,
                        "gold": ", ".join(gold[:5]), "cited": ", ".join(cited[:8])})
    return out if isinstance(out, JudgeScore) else JudgeScore(**dict(out))


# 채점 대상 20문항. 라벨링 가이드의 "정답 있음 20"(조항번호형 8 · 상품명조건형 6 · 표수치형 6).
# 미답변형·적대적은 거절이 정답이라 이 루브릭이 아니라 미답변 정확도와 적대적 채점으로 본다.
JUDGE_SET = [f"S{i:02d}" for i in list(range(1, 15)) + list(range(20, 26))]
ITEMS = ("basis_match", "numeric_ok", "no_extra_claim")


def score_result(result: dict, golden: list[dict], gold_ids: dict[str, list[str]],
                 ids: list[str] | None = None, verbose: bool = False) -> dict:
    """결과 파일의 답변을 루브릭으로 채점한다. 거절한 문항은 0점이 아니라 '채점 제외'로 둔다.

    거절은 답변 품질이 아니라 오거절률이 보는 것이다. 여기서 0점을 주면 두 지표가 같은 실수를
    두 번 센다.
    """
    ids = ids or JUDGE_SET
    by_id = {r["id"]: r for r in result.get("rows", [])}
    rows = []
    for item in golden:
        if item["id"] not in ids:
            continue
        row = by_id.get(item["id"])
        if row is None:
            continue
        answered = bool(row.get("answered")) and bool(row.get("answer"))
        cited = row.get("chunk_ids") or ([row["top_chunk"]] if row.get("top_chunk") else [])
        out = {"id": item["id"], "type": item.get("type"), "answered": answered}
        if answered:
            sc = score_one(item["question"], row["answer"], item.get("answer", ""),
                           gold_ids.get(item["id"], []), cited)
            out.update({k: getattr(sc, k) for k in ITEMS})
            out["total"] = sc.total
            out["reason"] = sc.reason
        rows.append(out)
        if verbose:
            mark = f"{out['total']}/6  {out['basis_match']}·{out['numeric_ok']}·{out['no_extra_claim']}" if answered else "거절(채점 제외)"
            print(f"  {item['id']} {str(item.get('type', ''))[:6]:8} {mark}  {out.get('reason', '')[:50]}")
    scored = [r for r in rows if r["answered"]]
    n = len(scored) or 1

    def mean(key: str, subset: list[dict]) -> float:
        return round(sum(r[key] for r in subset) / (len(subset) or 1), 3)

    by_type: dict[str, list[dict]] = {}
    for r in scored:
        by_type.setdefault(r["type"], []).append(r)
    return {"n": len(rows), "n_scored": len(scored), "n_refused": len(rows) - len(scored),
            "mean_total": mean("total", scored),
            "mean_by_item": {k: mean(k, scored) for k in ITEMS},
            "full_marks": sum(1 for r in scored if r["total"] == 6),
            "by_type": {t: {"n": len(rs), "mean_total": mean("total", rs),
                            **{k: mean(k, rs) for k in ITEMS}} for t, rs in sorted(by_type.items())},
            "rows": rows}


def load_calibration(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def calibrate(calib: list[dict], verbose: bool = False) -> dict:
    """강사 답변 20개를 judge 로 채점해 강사 라벨과 맞춰 본다.

    같은 답변을 같은 루브릭으로 사람과 모델이 각각 채점한 것이라, 차이는 전부 judge 의 몫이다.
    """
    rows = []
    for r in calib:
        sc = score_one(r["question"], r["answer"], r.get("reference", ""), r.get("gold", []),
                       r.get("cited", []))
        h = r["human"]
        row = {"id": r["id"], "judge": {k: getattr(sc, k) for k in ITEMS},
               "human": {k: h[k] for k in ITEMS}, "judge_total": sc.total,
               "human_total": sum(h[k] for k in ITEMS), "reason": sc.reason}
        rows.append(row)
        if verbose:
            d = row["judge_total"] - row["human_total"]
            print(f"  {r['id']}  judge {row['judge_total']}/6  사람 {row['human_total']}/6  "
                  f"{'=' if d == 0 else f'{d:+d}'}  {sc.reason[:50]}")
    try:
        agree = {"total": agreement([r["judge_total"] for r in rows], [r["human_total"] for r in rows])}
        for k in ITEMS:
            agree[k] = agreement([r["judge"][k] for r in rows], [r["human"][k] for r in rows])
    except NotImplementedError:
        # 실습 2 전. 채점 20번은 끝났으니 버리지 않고 문항별 비교만 돌려준다. agreement 를 채운 뒤 다시 돌리면 일치도가 붙는다.
        agree = None
    bias = round(sum(r["judge_total"] - r["human_total"] for r in rows) / (len(rows) or 1), 3)
    return {"n": len(rows), "agreement": agree, "judge_minus_human_avg": bias, "rows": rows}


def agreement(judge_scores: list[int], human_labels: list[int]) -> dict:
    """judge 와 사람 라벨의 일치도. 완전 일치와 ±1 이내를 나눠 본다.

    반환: {"n", "exact", "within_1", "mae"}. 둘의 길이가 다르면 짧은 쪽까지만 본다.
    둘 다 비어 있으면 {"n": 0}. exact · within_1 은 비율(소수 3자리), mae 는 평균 절대 오차(소수 3자리).

    [6회차 실습 2] 이 함수를 채우세요. 테스트는 tests/test_judge.py 입니다.
    왜 셋인가: 완전 일치만 보면 6점 척도에서 한 점 차이도 불일치가 되어 judge 가 쓸모없어 보이고,
    ±1 만 보면 늘 후하게 주는 judge 를 못 잡는다. MAE 가 그 치우침의 크기다.
    """
    pairs = list(zip(judge_scores, human_labels))   # 길이가 다르면 짧은 쪽까지만
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    diffs = [abs(int(j) - int(h)) for j, h in pairs]
    return {"n": n,
            "exact": round(sum(1 for d in diffs if d == 0) / n, 3),
            "within_1": round(sum(1 for d in diffs if d <= 1) / n, 3),
            "mae": round(sum(diffs) / n, 3)}
