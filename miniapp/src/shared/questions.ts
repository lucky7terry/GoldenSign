/**
 * 문진 모드의 질문 목록. 문구는 확정 시안 그대로이며 순서가 화면의 질문
 * 번호(1~4)다. 질문 전환 시 질문 문장 TTS 재생은 이번 범위 밖이라 없다.
 */

export const QUESTIONS: readonly string[] = [
  "어디가 어떻게 아프세요?",
  "알레르기가 있으세요?",
  "드시는 약 있으세요?",
  "무슨 일이 있으셨어요?",
]

export const QUESTION_COUNT = QUESTIONS.length
