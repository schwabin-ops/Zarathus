from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# ----------------------------- Configuration -----------------------------
st.set_page_config(
    page_title="Aegis | Financial Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(120,130,145,.08); border: 1px solid rgba(120,130,145,.18); padding: 14px 16px; border-radius: 12px;}
.small-note {font-size: .86rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)


# ----------------------------- Finance engine -----------------------------
def weighted_portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    columns = [c for c in returns.columns if c in weights]
    if not columns:
        return pd.Series(dtype=float)
    w = pd.Series({c: weights[c] for c in columns}, dtype=float)
    if w.sum() <= 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    return returns[columns].fillna(0.0).mul(w, axis=1).sum(axis=1).rename("portfolio_return")


def portfolio_metrics(port_returns: pd.Series) -> dict[str, float]:
    r = pd.Series(port_returns).dropna().astype(float)
    if len(r) < 2:
        return {
            "annual_return": 0.0, "annual_volatility": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "var_95": 0.0, "es_95": 0.0,
        }
    equity = (1 + r).cumprod()
    annual_return = float(equity.iloc[-1] ** (252 / len(r)) - 1) if equity.iloc[-1] > 0 else -1.0
    annual_vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = float((r.mean() * 252) / annual_vol) if annual_vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    losses = -r
    var95 = float(losses.quantile(0.95))
    tail = losses[losses >= var95]
    es95 = float(tail.mean()) if len(tail) else var95
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "var_95": var95,
        "es_95": es95,
    }


def moving_average_backtest(close: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    close = pd.Series(close).dropna().astype(float)
    if len(close) <= slow + 2 or fast >= slow or fast < 1:
        return pd.DataFrame()
    df = pd.DataFrame({"close": close})
    df["fast_ma"] = df["close"].rolling(fast).mean()
    df["slow_ma"] = df["close"].rolling(slow).mean()
    df["signal"] = (df["fast_ma"] > df["slow_ma"]).astype(float)
    # Today's signal is applied to the following period to reduce look-ahead bias.
    df["position"] = df["signal"].shift(1).fillna(0.0)
    df["asset_return"] = df["close"].pct_change().fillna(0.0)
    df["strategy_return"] = df["position"] * df["asset_return"]
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = (1 + df["asset_return"]).cumprod()
    return df.dropna(subset=["fast_ma", "slow_ma"])


def dcf_valuation(base_fcf: float, growth: float, wacc: float,
                  terminal_growth: float, years: int, net_debt: float = 0.0) -> dict:
    if wacc <= terminal_growth:
        raise ValueError("WACC는 영구 성장률보다 커야 합니다.")
    cashflows = []
    pv_sum = 0.0
    for year in range(1, years + 1):
        fcf = base_fcf * ((1 + growth) ** year)
        pv = fcf / ((1 + wacc) ** year)
        cashflows.append({"연도": year, "예상 FCF": fcf, "현재가치": pv})
        pv_sum += pv
    terminal_fcf = base_fcf * ((1 + growth) ** years) * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    ev = pv_sum + pv_terminal
    return {
        "enterprise_value": ev,
        "equity_value": ev - net_debt,
        "terminal_value_share": pv_terminal / ev if ev else 0.0,
        "terminal_value": terminal_value,
        "cashflows": cashflows,
    }


# ----------------------------- Data provider -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.rename_axis("Date").dropna(subset=["Close"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_company_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).get_info()
        keys = ["longName", "industry", "sector", "longBusinessSummary", "marketCap", "currency"]
        return {k: info.get(k) for k in keys if info.get(k) is not None}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def get_latest_news(ticker: str, limit: int = 5) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news or []
        result = []
        for item in raw[:limit]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            canonical = content.get("canonicalUrl")
            link = canonical.get("url") if isinstance(canonical, dict) else content.get("link")
            if not link:
                link = item.get("link")
            provider = content.get("provider", {})
            publisher = provider.get("displayName", "") if isinstance(provider, dict) else item.get("publisher", "")
            if title:
                result.append({"title": title, "link": link, "publisher": publisher})
        return result
    except Exception:
        return []


# ----------------------------- Ollama client -----------------------------
def read_secret(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        return default


def ollama_is_configured() -> bool:
    return bool(read_secret("OLLAMA_BASE_URL") and read_secret("OLLAMA_MODEL"))


def ollama_chat(prompt: str, context: dict) -> str:
    base_url = read_secret("OLLAMA_BASE_URL").rstrip("/")
    model = read_secret("OLLAMA_MODEL")
    api_key = read_secret("OLLAMA_API_KEY")
    if not base_url or not model:
        raise RuntimeError("Streamlit Secrets에 OLLAMA_BASE_URL과 OLLAMA_MODEL을 설정하세요.")
    url = base_url + ("/chat" if base_url.endswith("/api") else "/api/chat")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    system = (
        "You are a cautious financial research assistant. Use supplied context for current figures. "
        "Do not fabricate data, prices, or sources. Separate facts, assumptions, and unknowns. "
        "Explain downside risks and uncertainty. Do not give personalized financial advice. "
        "Answer in Korean unless the user asks otherwise."
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "분석 데이터(JSON):\n"
             + json.dumps(context, ensure_ascii=False, default=str)
             + "\n\n요청:\n" + prompt},
        ],
        "options": {"temperature": 0.2},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=(10, 180))
    if response.status_code >= 400:
        raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:500]}")
    result = response.json().get("message", {}).get("content")
    if not result:
        raise RuntimeError("Ollama 응답에 message.content가 없습니다.")
    return result


# ----------------------------- UI -----------------------------
st.title("🛡️ Aegis Financial Intelligence")
st.caption("통합 금융 데이터 · 포트폴리오 위험 · 퀀트 백테스트 · DCF · Ollama 금융 AI")

with st.sidebar:
    st.header("분석 설정")
    ticker_input = st.text_input("관심 종목", "AAPL, MSFT, NVDA")
    tickers = list(dict.fromkeys([x.strip().upper() for x in ticker_input.split(",") if x.strip()]))[:15]
    period = st.selectbox("조회 기간", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    st.divider()
    st.subheader("Ollama AI")
    st.write("설정 상태:", "설정됨" if ollama_is_configured() else "Secrets 설정 필요")
    st.caption("API 키는 GitHub 코드에 넣지 말고 Streamlit Secrets에 저장하세요.")

if not tickers:
    st.warning("분석할 티커를 하나 이상 입력해 주세요.")
    st.stop()

price_data, load_errors = {}, {}
with st.spinner("시장 데이터를 가져오는 중..."):
    for symbol in tickers:
        try:
            df = get_price_history(symbol, period)
            if df is not None and not df.empty:
                price_data[symbol] = df
            else:
                load_errors[symbol] = "가격 데이터가 비어 있습니다."
        except Exception as exc:
            load_errors[symbol] = str(exc)

if load_errors:
    with st.expander("데이터 조회 경고"):
        for symbol, error in load_errors.items():
            st.write(f"**{symbol}** — {error}")

available = list(price_data)
if not available:
    st.error("가격 데이터를 가져오지 못했습니다. 티커와 네트워크를 확인하세요.")
    st.stop()

returns = pd.DataFrame({s: df["Close"].pct_change() for s, df in price_data.items()}).dropna(how="all")
prices = pd.DataFrame({s: df["Close"] for s, df in price_data.items()}).dropna(how="all")
tabs = st.tabs(["시장 대시보드", "포트폴리오 위험", "퀀트 백테스트", "기업가치(DCF)", "AI 애널리스트", "데이터"])

with tabs[0]:
    st.subheader("시장 현황")
    metric_cols = st.columns(min(4, len(available)))
    for i, symbol in enumerate(available[:4]):
        df = price_data[symbol]
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        metric_cols[i].metric(symbol, f"{last:,.2f}", f"{((last / prev) - 1) * 100:+.2f}%" if prev else "—")
    normalized = prices / prices.iloc[0] * 100
    st.plotly_chart(px.line(normalized, x=normalized.index, y=normalized.columns,
                            title="상대 성과 (시작값 = 100)"), use_container_width=True)
    selected = st.selectbox("종목 상세", available)
    df = price_data[selected]
    info = get_company_info(selected)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최근 종가", f"{float(df['Close'].iloc[-1]):,.2f}")
    c2.metric("기간 최고가", f"{float(df['High'].max()):,.2f}")
    c3.metric("기간 최저가", f"{float(df['Low'].min()):,.2f}")
    c4.metric("관측치", f"{len(df):,}일")
    if info:
        st.write(f"**회사:** {info.get('longName') or selected} · **업종:** {info.get('industry') or '정보 없음'}")
        if info.get("longBusinessSummary"):
            with st.expander("회사 개요"):
                st.write(info["longBusinessSummary"])
    news = get_latest_news(selected)
    if news:
        st.markdown("**최근 뉴스 헤드라인**")
        for item in news:
            title, link, publisher = item["title"], item.get("link"), item.get("publisher", "")
            st.markdown(f"- [{title}]({link}) — {publisher}" if link else f"- {title} — {publisher}")

with tabs[1]:
    st.subheader("포트폴리오 위험 분석")
    st.write("투자 비중을 입력하세요. 합계가 100%가 아니면 자동 정규화합니다.")
    weight_cols = st.columns(min(4, len(available)))
    weights = {}
    for i, symbol in enumerate(available):
        default = 100.0 / len(available)
        weights[symbol] = weight_cols[i % len(weight_cols)].number_input(
            f"{symbol} 비중 (%)", min_value=0.0, max_value=100.0,
            value=round(default, 2), step=1.0, key=f"weight_{symbol}"
        ) / 100
    if sum(weights.values()) <= 0:
        st.warning("비중 합계가 0입니다.")
        port_returns = pd.Series(dtype=float)
        metrics = None
    else:
        weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        port_returns = weighted_portfolio_returns(returns, weights)
        metrics = portfolio_metrics(port_returns)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("연환산 수익률(근사)", f"{metrics['annual_return']:.2%}")
        c2.metric("연환산 변동성", f"{metrics['annual_volatility']:.2%}")
        c3.metric("샤프 비율(무위험수익률 0 가정)", f"{metrics['sharpe']:.2f}")
        c4.metric("최대 낙폭", f"{metrics['max_drawdown']:.2%}")
        c5, c6 = st.columns(2)
        c5.metric("95% 일간 VaR", f"{metrics['var_95']:.2%}")
        c6.metric("95% 일간 ES", f"{metrics['es_95']:.2%}")
        st.plotly_chart(px.line((1 + port_returns.fillna(0)).cumprod(),
                                title="포트폴리오 누적 가치 (초기값 1.0)"), use_container_width=True)
        if len(returns.columns) > 1:
            st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1, title="자산 수익률 상관관계"), use_container_width=True)
        st.caption("과거 수익률 기반 추정치이며 미래 손실을 보장하거나 예측하지 않습니다.")

with tabs[2]:
    st.subheader("이동평균 교차 전략 백테스트")
    selected_bt = st.selectbox("백테스트 종목", available, key="bt_symbol")
    fast = st.slider("단기 이동평균", 3, 60, 10)
    slow = st.slider("장기 이동평균", 10, 200, 40)
    if fast >= slow:
        st.warning("단기 이동평균 기간은 장기 이동평균보다 작아야 합니다.")
    else:
        bt = moving_average_backtest(price_data[selected_bt]["Close"], fast, slow)
        if bt.empty:
            st.info("백테스트를 수행하기에 데이터가 부족합니다.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("전략 누적 수익률", f"{float(bt['strategy_equity'].iloc[-1] - 1):.2%}")
            c2.metric("매수 후 보유", f"{float(bt['buy_hold_equity'].iloc[-1] - 1):.2%}")
            c3.metric("포지션 변경 횟수", f"{int(bt['position'].diff().abs().fillna(0).sum())}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt.index, y=bt["strategy_equity"], name="전략"))
            fig.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="Buy & Hold"))
            fig.update_layout(title="초기 자산 1.0 기준", xaxis_title="", yaxis_title="자산 가치")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("교육용 단순 백테스트입니다. 거래비용·세금·슬리피지·체결 제약을 충분히 반영하지 않습니다.")

with tabs[3]:
    st.subheader("DCF 기업가치 시나리오")
    st.write("재무 수치는 사용자가 입력해야 하며 자동으로 공시 실측값이라고 간주하지 않습니다.")
    col1, col2, col3 = st.columns(3)
    with col1:
        fcf = st.number_input("기준 연도 FCF (입력 통화 단위)", value=1_000_000_000.0, step=100_000_000.0, format="%.0f")
        growth = st.number_input("연간 FCF 성장률 (%)", min_value=-90.0, max_value=200.0, value=8.0) / 100
    with col2:
        wacc = st.number_input("WACC (%)", min_value=0.1, max_value=100.0, value=9.0) / 100
        terminal_growth = st.number_input("영구 성장률 (%)", min_value=-50.0, max_value=30.0, value=2.5) / 100
    with col3:
        years = st.slider("예측 기간(년)", 3, 10, 5)
        net_debt = st.number_input("순부채 (순현금이면 음수)", value=0.0, step=100_000_000.0, format="%.0f")
    if terminal_growth >= wacc:
        st.error("영구 성장률은 WACC보다 낮아야 합니다.")
    else:
        result = dcf_valuation(fcf, growth, wacc, terminal_growth, years, net_debt)
        a, b, c = st.columns(3)
        a.metric("추정 기업가치(EV)", f"{result['enterprise_value']:,.0f}")
        b.metric("추정 자기자본 가치", f"{result['equity_value']:,.0f}")
        c.metric("최종가치 비중", f"{result['terminal_value_share']:.1%}")
        st.dataframe(pd.DataFrame(result["cashflows"]), use_container_width=True, hide_index=True)
        st.caption("DCF 결과는 가정에 민감하며 투자 추천이나 확정 가치가 아닙니다.")

with tabs[4]:
    st.subheader("Ollama 금융 AI 애널리스트")
    st.write("기업 분석, 위험 요인, 전략 설명, 재무 데이터 해석을 자연어로 요청할 수 있습니다.")
    if not ollama_is_configured():
        st.warning("Ollama API 설정이 없습니다. README의 Streamlit Secrets 안내를 따라주세요.")
    prompt = st.text_area(
        "분석 요청",
        value=f"{', '.join(available)}의 최근 시장 데이터를 바탕으로 분석해줘. 확인된 수치와 가정을 구분하고 위험 요인 및 추가로 확인할 자료를 알려줘.",
        height=150
    )
    context = {
        "tickers": available,
        "period": period,
        "latest_prices": {s: round(float(price_data[s]["Close"].iloc[-1]), 4) for s in available},
        "period_return": {
            s: round(float(price_data[s]["Close"].iloc[-1] / price_data[s]["Close"].iloc[0] - 1), 4)
            for s in available
        },
        "guardrails": [
            "Do not invent current financial figures or sources.",
            "Distinguish supplied data, assumptions, and unknowns.",
            "Explain uncertainty and downside risks.",
            "Do not present output as personalized financial advice."
        ],
    }
    if st.button("AI 분석 실행", type="primary", disabled=not ollama_is_configured()):
        with st.spinner("Ollama 모델이 분석 중..."):
            try:
                st.markdown(ollama_chat(prompt, context))
            except Exception as exc:
                st.error(f"AI 요청 실패: {exc}")
    st.caption("LLM은 오류를 낼 수 있습니다. 숫자 계산은 정량 엔진과 원천 데이터를 기준으로 검토하세요.")

with tabs[5]:
    st.subheader("가격 데이터")
    st.dataframe(prices.tail(30).sort_index(ascending=False), use_container_width=True)
    st.download_button("가격 데이터 CSV 다운로드",
                       data=prices.to_csv(index_label="Date").encode("utf-8-sig"),
                       file_name="market_prices.csv", mime="text/csv")
    st.markdown("**데이터 품질 점검**")
    st.write({
        "요청 티커": tickers,
        "정상 조회 티커": available,
        "가격 데이터 행 수": int(len(prices)),
        "결측값 수": int(prices.isna().sum().sum()),
        "마지막 관측일": str(prices.index.max().date()) if not prices.empty else None,
    })

st.divider()
st.markdown(
    '<div class="small-note">Aegis는 교육·연구용 분석 도구입니다. 기관용 Aladdin의 전체 기능, '
    '데이터 계약, 운영 통제 및 검증 체계를 대체하지 않습니다. 투자 결정 전 원천 데이터와 계산을 확인하세요.</div>',
    unsafe_allow_html=True
)
