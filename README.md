# Aegis Financial Intelligence (Flat / no folders)

모든 앱 코드는 단일 `main.py`에 들어 있습니다. GitHub 저장소 루트에는 `main.py`, `requirements.txt`, `README.md`, `LICENSE`, `.gitignore`만 올리면 됩니다. 별도의 `core/` 또는 `.streamlit/` 폴더는 필요하지 않습니다.

## Streamlit Cloud 배포
1. 이 ZIP의 파일을 압축 해제합니다.
2. 파일 5개를 GitHub 저장소의 최상위(루트)에 업로드합니다.
3. Streamlit Community Cloud에서 저장소를 연결하고 Main file path를 `main.py`로 지정합니다.
4. 앱 설정의 Settings → Secrets에 아래 TOML을 입력합니다.

```toml
OLLAMA_BASE_URL = "https://ollama.com"
OLLAMA_MODEL = "YOUR_AVAILABLE_OLLAMA_MODEL"
OLLAMA_API_KEY = "YOUR_OLLAMA_API_KEY"
```

모델명과 키는 사용 중인 Ollama 서비스에서 확인한 실제 값으로 바꾸세요. 자체 호스팅 Ollama 서버가 인증을 요구하지 않으면 API 키를 생략할 수 있습니다. `OLLAMA_BASE_URL`에는 API 루트 주소를 넣습니다. 이 앱은 `/api/chat` 경로를 붙이며 주소가 `/api`로 끝나면 `/chat`을 붙입니다.

**중요:** Streamlit Secrets는 키를 저장할 뿐 모델을 실행해 주지 않습니다. Streamlit Cloud에서 접속 가능한 Ollama Cloud 또는 외부 Ollama 서버가 필요합니다. `localhost`는 배포된 Streamlit 서버 내부를 가리키므로 개인 컴퓨터의 Ollama에 연결되지 않습니다. 외부 서버에는 HTTPS, 인증 및 접근 제한을 적용하세요.

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run main.py
```

로컬에서 Ollama를 쓰려면 `.streamlit/secrets.toml`을 만들 수 있지만, 이 파일은 GitHub에 올리지 마세요. 또는 Streamlit 실행 환경의 secrets 설정을 이용하세요.

## 포함 기능
- yfinance 기반 주가, 기업 개요, 뉴스 조회
- 상대 성과 차트
- 포트폴리오 연환산 수익률·변동성·샤프·최대 낙폭·역사적 VaR/ES
- 이동평균 교차 전략의 교육용 백테스트
- 사용자 입력 기반 DCF 가치 시나리오
- Ollama를 통한 한국어 자연어 분석
- CSV 다운로드

## 한계
교육·연구용 예제이며 기관용 Aladdin을 대체하지 않습니다. 시장 데이터는 지연/누락/수정될 수 있습니다. 백테스트는 거래비용, 세금, 슬리피지 및 체결 제약을 충분히 반영하지 않습니다. LLM 답변은 틀릴 수 있으므로 공시·원천 데이터와 계산 결과를 확인하세요. 투자 주문을 실행하지 않습니다.

이 코드는 업로드된 오픈소스 프로젝트의 구조적 아이디어를 참고해 새로 통합한 예제이며, 해당 프로젝트들의 소스 전체를 그대로 복사한 것은 아닙니다. upstream 코드를 추가할 경우 각 저장소와 파일의 라이선스를 따로 확인해야 합니다.
