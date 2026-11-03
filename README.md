# Aegis Financial Intelligence — Gemini Edition

모든 앱 코드는 단일 `main.py`에 들어 있습니다. GitHub 저장소 루트에 `main.py`, `requirements.txt`, `README.md`, `LICENSE`, `.gitignore`, `.env` 6개 파일을 올리면 됩니다.

## 이번 변경점
### Yahoo Finance 전 종목 지원
- 사이드바 **🔎 Yahoo Finance 종목 검색**: 회사명·티커·한국 6자리 코드로 검색해 결과에서 골라 추가합니다. 주식·ETF·지수·펀드·암호화폐·환율·선물 등 Yahoo Finance에서 조회되는 모든 심볼을 찾을 수 있습니다.
- **관심 종목** 칸에 티커를 직접 입력할 수도 있습니다 (예: `AAPL`, `005930.KS`, `035720.KQ`, `^KS11`, `BTC-USD`, `KRW=X`, `GC=F`). 6자리 한국 코드만 넣으면 `.KS`/`.KQ`를 자동으로 붙입니다.
- **프리셋**: 미국 빅테크, 한국 대형주, 주요 지수, 대표 ETF, 암호화폐, 환율·원자재를 한 번에 추가.
- 종목 수 제한(기존 15개)을 없앴고, 시장 탭에 모든 종목 지표를 격자로 표시합니다. 30개를 넘으면 속도 안내만 표시합니다.
- 시나리오 예측 탭에서 **Gemini 분석 대상 종목**을 따로 고를 수 있어 종목이 많아도 요청을 가볍게 유지할 수 있습니다.
- 조회 기간에 `max` 추가.

### API 키를 `.env`로 관리
- `.env` 파일을 저장소에 함께 올리고, 앱이 `python-dotenv`로 읽습니다. `.gitignore`에서 `.env`를 뺐습니다.
- 읽는 순서: 환경변수(`.env` 포함) → Streamlit Secrets. 기존 Secrets 설정도 계속 동작합니다.

## 설정 (`.env`)
```env
GEMINI_API_KEY=여기에_발급받은_키
GEMINI_MODEL=gemini-flash-latest
GEMINI_THINKING_LEVEL=high
```
- `GOOGLE_API_KEY`라는 이름도 인식합니다.
- `GEMINI_MODEL`을 생략하면 `gemini-flash-latest`를 씁니다. 사이드바에서도 바꿀 수 있습니다.

> **주의:** 공개(public) 저장소에 키를 올리면 Google이 키를 '유출됨'으로 자동 차단해 `Your API key was reported as leaked` 오류가 날 수 있고, 다른 사람이 키를 써서 할당량이 소진될 수 있습니다. 저장소를 **비공개(private)** 로 두는 것을 권장합니다. 차단되면 새 키를 발급해 `.env`만 바꾸면 됩니다.

## Streamlit Cloud 배포
1. 파일 6개를 GitHub 저장소 루트에 업로드합니다. (`.env`는 점으로 시작하는 숨김 파일이라 탐색기에서 안 보일 수 있습니다. GitHub에서 **Add file → Create new file**로 이름을 `.env`로 직접 만들어도 됩니다.)
2. Streamlit Community Cloud에서 저장소를 연결하고 Main file path를 `main.py`로 지정합니다.
3. Secrets 설정은 필요 없습니다.

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run main.py
```

## 기존 기능
- **Gemini 사고 모드**: 사고 수준(minimal/low/medium/high), 사고 요약 실시간 스트리밍, 토큰 사용량 표시. 3.x는 `thinking_level`, 2.5는 `thinking_budget` 자동 사용.
- **Google 뉴스**: 종목별 Google News RSS 병렬 수집 (한국어/영어, 기간·기사 수 조절, CSV 다운로드)
- **뉴스 반영 시나리오 예측**: Gemini가 헤드라인·시장 통계로 강세/기준/약세 확률, drift 조정, 변동성 배수, 꼬리 이벤트를 JSON으로 산출 → Student-t + 포아송 점프 몬테카를로. 키가 없으면 규칙 기반 감성으로 대체.
- **AI 애널리스트**: 멀티턴 대화, 시장 통계·뉴스·시나리오 결과를 컨텍스트로 사용
- 포트폴리오 위험(소르티노·칼마·베타·알파·추적오차·위험 기여도), 이동평균 백테스트 + 그리드 탐색, DCF + 민감도 히트맵
- yfinance 일괄 다운로드, 병렬 수집, 캐시, `st.fragment` 탭 분리, numpy 벡터화

## 한계
교육·연구용 예제입니다. 시장 데이터는 지연·누락·수정될 수 있고, Yahoo Finance 검색과 Google News RSS는 공식 API가 아니므로 형식 변경이나 요청 제한이 있을 수 있습니다. 종목을 매우 많이 추가하면 요청 제한에 걸릴 수 있습니다. 통화가 다른 종목을 섞은 포트폴리오는 환율 효과를 반영하지 않습니다. 뉴스 해석과 시나리오 파라미터는 LLM 판단이라 틀릴 수 있으며, 몬테카를로 결과는 확률 모형의 출력이지 가격 예측 보장이 아닙니다. 투자 주문을 실행하지 않으며 개인 맞춤 투자 자문이 아닙니다.
