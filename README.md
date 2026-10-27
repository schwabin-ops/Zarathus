# Aegis Financial Intelligence

Streamlit Cloud에서 배포할 수 있도록 구성한 교육·연구용 통합 금융 분석 앱입니다. OpenBB의 데이터 통합 접근, FinRobot의 재무 분석 파이프라인, Qlib/FinRL의 퀀트 연구 관점, ORE의 위험 관리 관점을 참고해 독립적으로 구현했습니다. 업로드된 프로젝트 소스를 대량 복사하지 않으며, 이 앱은 해당 프로젝트들의 공식 구성요소를 직접 포함하지 않습니다.

## 기능

- Yahoo Finance 기반 가격·기업 개요·뉴스 헤드라인 조회 (`yfinance`)
- 상대 성과 차트와 시장 대시보드
- 포트폴리오 가중 수익률, 연환산 변동성, 샤프 비율, 최대 낙폭, 역사적 VaR/ES
- 이동평균 교차 전략의 간단한 백테스트
- 사용자 입력 기반 DCF 가치 평가
- Ollama API를 통한 한국어 자연어 금융 분석
- 가격 데이터 CSV 다운로드

## Streamlit Cloud 배포

1. 이 폴더의 파일을 GitHub 저장소 루트에 올립니다.
2. Streamlit Community Cloud에서 해당 저장소를 연결합니다.
3. Main file path를 `main.py`로 지정합니다.
4. 앱의 **Settings → Secrets**에 아래 설정을 입력합니다.

```toml
OLLAMA_BASE_URL = "https://ollama.com"
OLLAMA_MODEL = "YOUR_AVAILABLE_OLLAMA_MODEL"
OLLAMA_API_KEY = "YOUR_OLLAMA_API_KEY"
```

`OLLAMA_MODEL`은 사용 중인 Ollama 계정/엔드포인트에서 실제로 사용 가능한 모델명으로 바꾸세요. Ollama 서버가 API 키를 요구하지 않는 자체 호스팅 환경이면 `OLLAMA_API_KEY`는 생략할 수 있습니다. `OLLAMA_BASE_URL`은 API의 루트 주소여야 합니다. 이 앱은 루트 주소 뒤에 `/api/chat`을 붙입니다. 주소가 `/api`로 끝나면 `/chat`을 붙입니다.

**중요:** Streamlit Secrets는 API 키를 저장할 뿐 Ollama 모델을 호스팅하지 않습니다. Streamlit Cloud에서 접근 가능한 Ollama Cloud 또는 외부 Ollama 서버가 필요합니다. 로컬 PC의 `localhost` 주소는 Streamlit Cloud에서 사용자의 PC를 가리키지 않습니다. 외부 서버를 쓸 때는 HTTPS, 인증, 요청 제한, 접근 제어를 설정하세요.

## 로컬 실행

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

`.streamlit/secrets.toml.example`을 참고해 `.streamlit/secrets.toml`을 만들고 실제 값을 넣으세요. 해당 실제 파일은 Git에 커밋하지 마세요.

## 계산 정의

- 일간 수익률: `P_t / P_(t-1) - 1`
- 연환산 변동성: `std(daily_returns) * sqrt(252)`
- 샤프 비율: `mean(daily_returns) * 252 / annualized_volatility` (현재 구현은 무위험 수익률을 0으로 가정)
- 최대 낙폭: `equity / running_peak - 1`의 최솟값
- 역사적 VaR 95%: 일간 손실 분포의 95% 분위수
- 역사적 ES 95%: 손실이 VaR 이상인 관측치의 평균
- DCF: 명시적 기간 FCF의 현재가치 합 + 할인된 Gordon Growth terminal value

## 한계와 안전

- 기관용 BlackRock Aladdin의 대체품이 아닙니다. 포지션/주문 관리, 기관 데이터 라이선스, 실시간 리스크, 규제 보고, 권한·감사 체계가 완비된 제품이 아닙니다.
- `yfinance` 데이터는 데이터 공급 계약을 대체하지 않으며 지연, 누락, 수정 가능성이 있습니다.
- 이 백테스트는 거래비용, 세금, 슬리피지, 배당 재투자 세부사항, 주문 체결 제약을 충분히 반영하지 않습니다.
- 과거 수익률은 미래 수익률을 보장하지 않습니다. 투자 판단에 앞서 원천 데이터와 계산 결과를 검증하세요.
- Ollama 출력은 환각 또는 잘못된 해석을 낼 수 있습니다. 정량 계산을 LLM에게 맡기지 말고 코드 결과와 공시를 확인하세요.
- 이 앱은 투자 주문을 실행하지 않으며 개인 맞춤 투자 자문을 제공하도록 설계되지 않았습니다.

## 라이선스

이 앱의 새로 작성된 코드에는 MIT License를 적용합니다. 각 upstream 프로젝트의 코드를 복사하거나 통합할 경우에는 해당 프로젝트와 각 파일의 라이선스 및 저작권 고지를 별도로 준수해야 합니다.
