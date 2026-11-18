/**
 * 답 단어들을 문장 후보로 조합하는 자리. 조합 규칙은 다음 브랜치에서 이 함수만
 * 채운다 — 지금은 항상 빈 배열을 돌려주는 스텁이고, UI 는 후보가 비어 있으면
 * "문장 후보" 영역 자체를 그리지 않는다.
 */

import type {AnswerWord} from "./channels"

export type Candidate = {
  /** 조합된 문장 전체. */
  text: string
  /** 이 문장을 만든 단어들의 id. 단어 삭제 시 후보를 무효화하는 근거다. */
  wordIds: number[]
}

export function buildCandidates(words: AnswerWord[], questionIndex: number): Candidate[] {
  void words
  void questionIndex
  return []
}