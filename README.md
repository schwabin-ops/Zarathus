# Aegis Financial Intelligence — Gemini Edition

모든 앱 코드는 단일 `main.py`에 들어 있습니다. GitHub 저장소 루트에 `main.py`, `requirements.txt`, `README.md`, `LICENSE`, `.gitignore` 5개 파일만 올리면 됩니다.

## 주요 변경점 (Ollama → Gemini)
- **Google Gemini API** (`google-genai` SDK)로 전환
- **사고(thinking) 시스템**: 사이드바에서 사고 수준(minimal / low / medium / high)을 고르고, 모델의 **사고 요약을 실시간 스트리밍**으로 볼 수 있습니다. Gemini 3.x 계열은 `thinking_level`, 2.5 계열은 `thinking_budget`을 자동으로 사용합니다. 응답마다 입력·사고·출력 토큰 수를 표시합니다.
- **Google 뉴스**: 종목별 Google News RSS를 병렬 수집해 시장 탭·뉴스 탭에 표시 (한국어/영어, 기간·기사 수 조절, CSV 다운로드)
- **뉴스 반영 시나리오 예측**: Gemini가 뉴스 헤드라인과 시장 통계를 읽고 JSON 스키마(구조화 출력)로
  강세/기준/약세 확률, 연 기대수익 조정, 변동성 배수, 꼬리 이벤트(확률·충격), 근거 헤드라인을 산출 →
  **몬테카를로 엔진**(Student-t 두꺼운 꼬리 + 포아송 점프 + 시나리오 혼합)에 그대로 입력합니다.
  LLM 출력은 안전 범위로 클램프되고 확률은 정규화되며, 모든 파라미터는 슬라이더로 직접 조정할 수 있습니다.
  API 키가 없으면 규칙 기반 감성 점수로 대체합니다.
- **AI 애널리스트**: 멀티턴 대화 + 시장 통계·뉴스·시나리오 결과를 컨텍스트로 사용

## 성능 개선
- yfinance **일괄 다운로드**(멀티스레드) — 종목 수만큼 요청하던 방식 → 1회 요청 (실패 종목만 개별 재시도)
- 기업 정보·뉴스 **병렬 수집**(ThreadPoolExecutor) + 캐시(가격 15분, 뉴스 10분, 기업정보 24시간)
- 각 탭을 `st.fragment`로 분리 → 슬라이더를 움직여도 해당 탭만 다시 계산
- 지표·백테스트·DCF·몬테카를로를 numpy **벡터화** (2만 경로 × 126일 시뮬레이션 약 0.5초)
- Gemini 클라이언트 재사용(`st.cache_resource`), 시나리오 결과를 세션에 보관하고 뉴스가 바뀌면 재생성 안내

## 추가 분석 기능
- 포트폴리오: 소르티노·칼마·베타·젠센 알파·추적오차(벤치마크 대비), 낙폭·롤링 변동성 차트, 위험 기여도
- 백테스트: 거래비용(bps) 반영, 이동평균 파라미터 그리드 탐색 히트맵
- DCF: yfinance 값으로 기본값 채움, 주당 내재가치, WACC × 영구성장률 민감도 히트맵
- 시장: 캔들 차트 + 이동평균

## Streamlit Cloud 배포
1. 파일 5개를 GitHub 저장소 루트에 업로드합니다.
2. Streamlit Community Cloud에서 저장소를 연결하고 Main file path를 `main.py`로 지정합니다.
3. 앱 Settings → Secrets에 아래 TOML을 입력합니다.

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"       # Google AI Studio에서 발급
GEMINI_MODEL = "gemini-flash-latest"         # 선택 사항
GEMINI_THINKING_LEVEL = "high"               # 선택 사항: minimal / low / medium / high
```

- `GEMINI_MODEL`을 생략하면 항상 최신 Flash 모델을 가리키는 `gemini-flash-latest`를 사용합니다. 특정 버전을 고정하려면 Google AI Studio의 모델 목록에서 확인한 모델 ID(예: `gemini-3.7-flash`)를 넣으세요. 사이드바에서도 바꿀 수 있습니다.
- `GOOGLE_API_KEY`라는 이름도 인식합니다.

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run main.py
```
로컬에서는 `.streamlit/secrets.toml`에 위 내용을 넣으면 됩니다(이 파일은 `.gitignore`에 포함되어 GitHub에 올라가지 않습니다).

## 한계
교육·연구용 예제입니다. 시장 데이터는 지연·누락·수정될 수 있고, Google News RSS는 공식 API가 아니므로 형식 변경이나 요청 제한이 있을 수 있습니다. 뉴스 해석과 시나리오 파라미터는 LLM 판단이라 틀릴 수 있으며, 몬테카를로 결과는 확률 모형의 출력이지 가격 예측 보장이 아닙니다. 백테스트는 세금·슬리피지·체결 제약을 반영하지 않습니다. 투자 주문을 실행하지 않으며 개인 맞춤 투자 자문이 아닙니다.
