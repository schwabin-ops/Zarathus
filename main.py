import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from core.data import get_price_history, get_company_info, get_latest_news
from core.finance import (
    portfolio_metrics, weighted_portfolio_returns, moving_average_backtest,
    dcf_valuation, safe_float
)
from core.llm import ollama_chat, ollama_is_configured

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

st.title("🛡️ Aegis Financial Intelligence")
st.caption("통합 금융 데이터 · 포트폴리오 위험 · 퀀트 백테스트 · 기업가치 분석 · Ollama 금융 AI")

with st.sidebar:
    st.header("분석 설정")
    ticker_input = st.text_input("관심 종목", "AAPL, MSFT, NVDA")
    tickers = list(dict.fromkeys([x.strip().upper() for x in ticker_input.split(",") if x.strip()]))[:15]
    period = st.selectbox("조회 기간", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    st.divider()
    st.subheader("Ollama AI")
    ai_ready = ollama_is_configured()
    st.write("API 설정:", "연결 설정됨" if ai_ready else "미설정")
    model_label = "Secrets에서 설정" if ai_ready else "AI 답변은 API 설정 후 사용 가능"
    st.caption(model_label)
    st.divider()
    st.caption("데이터 제공: Yahoo Finance(yfinance). 데이터 지연·누락·조정 방식이 있을 수 있습니다.")

if not tickers:
    st.warning("분석할 티커를 하나 이상 입력해 주세요.")
    st.stop()

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbols, lookback):
    data = {}
    errors = {}
    for symbol in symbols:
        try:
            df = get_price_history(symbol, lookback)
            if df is not None and not df.empty and "Close" in df:
                data[symbol] = df
            else:
                errors[symbol] = "가격 데이터가 비어 있습니다."
        except Exception as exc:
            errors[symbol] = str(exc)
    return data, errors

with st.spinner("시장 데이터를 가져오는 중..."):
    price_data, load_errors = load_prices(tuple(tickers), period)

if load_errors:
    with st.expander("데이터 조회 경고", expanded=False):
        for symbol, err in load_errors.items():
            st.write(f"**{symbol}** — {err}")

available = list(price_data.keys())
if not available:
    st.error("가격 데이터를 가져오지 못했습니다. 티커와 네트워크 상태를 확인해 주세요.")
    st.stop()

returns = pd.DataFrame({
    symbol: df["Close"].pct_change().rename(symbol)
    for symbol, df in price_data.items()
}).dropna(how="all")
prices = pd.DataFrame({
    symbol: df["Close"].rename(symbol)
    for symbol, df in price_data.items()
}).dropna(how="all")

tabs = st.tabs(["시장 대시보드", "포트폴리오 위험", "퀀트 백테스트", "기업가치(DCF)", "AI 애널리스트", "데이터"])

with tabs[0]:
    st.subheader("시장 현황")
    cols = st.columns(min(4, len(available)))
    for i, symbol in enumerate(available[:4]):
        df = price_data[symbol]
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        change = (last / prev - 1) * 100 if prev else 0
        cols[i].metric(symbol, f"{last:,.2f}", f"{change:+.2f}%")
    norm = prices / prices.iloc[0] * 100
    fig = px.line(norm, x=norm.index, y=norm.columns, title="기간별 상대 성과 (시작값 = 100)")
    fig.update_layout(legend_title_text="티커", xaxis_title="", yaxis_title="지수")
    st.plotly_chart(fig, use_container_width=True)
    selected = st.selectbox("종목 상세", available)
    df = price_data[selected]
    info = get_company_info(selected)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최근 종가", f"{float(df['Close'].iloc[-1]):,.2f}")
    c2.metric("기간 최고가", f"{float(df['High'].max()):,.2f}")
    c3.metric("기간 최저가", f"{float(df['Low'].min()):,.2f}")
    c4.metric("관측치", f"{len(df):,}일")
    if info:
        st.write(f"**회사:** {info.get('longName') or selected}  ·  **업종:** {info.get('industry') or '정보 없음'}")
        if info.get("longBusinessSummary"):
            with st.expander("회사 개요"):
                st.write(info["longBusinessSummary"])
    news = get_latest_news(selected, limit=5)
    if news:
        st.markdown("**최근 뉴스 헤드라인**")
        for item in news:
            title = item.get("title", "제목 없음")
            link = item.get("link")
            publisher = item.get("publisher", "")
            if link:
                st.markdown(f"- [{title}]({link}) — {publisher}")
            else:
                st.markdown(f"- {title} — {publisher}")

with tabs[1]:
    st.subheader("포트폴리오 위험 분석")
    st.write("티커별 투자 비중을 입력하세요. 비중 합계가 100%가 아니면 자동 정규화합니다.")
    default_weights = {s: 1 / len(available) for s in available}
    weight_cols = st.columns(min(4, len(available)))
    weights = {}
    for i, symbol in enumerate(available):
        weights[symbol] = weight_cols[i % len(weight_cols)].number_input(
            f"{symbol} 비중 (%)", min_value=0.0, max_value=100.0,
            value=round(default_weights[symbol] * 100, 2), step=1.0, key=f"weight_{symbol}"
        ) / 100
    if sum(weights.values()) <= 0:
        st.warning("비중 합계가 0입니다. 하나 이상의 비중을 입력하세요.")
    else:
        weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        port_returns = weighted_portfolio_returns(returns, weights)
        metrics = portfolio_metrics(port_returns)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("연환산 수익률(근사)", f"{metrics['annual_return']:.2%}")
        c2.metric("연환산 변동성", f"{metrics['annual_volatility']:.2%}")
        c3.metric("샤프 비율", f"{metrics['sharpe']:.2f}")
        c4.metric("최대 낙폭", f"{metrics['max_drawdown']:.2%}")
        c5, c6 = st.columns(2)
        c5.metric("95% 일간 VaR", f"{metrics['var_95']:.2%}")
        c6.metric("95% 일간 ES", f"{metrics['es_95']:.2%}")
        equity = (1 + port_returns.fillna(0)).cumprod()
        st.plotly_chart(px.line(equity, title="포트폴리오 누적 가치 (초기값 1.0)"), use_container_width=True)
        if len(returns.columns) > 1:
            st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1, title="자산 수익률 상관관계"), use_container_width=True)
        st.caption("위험 지표는 과거 일간 수익률을 이용한 표본 추정치이며 미래 손실을 보장하거나 예측하지 않습니다.")

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
            total = float(bt["strategy_equity"].iloc[-1] - 1)
            bh = float(bt["buy_hold_equity"].iloc[-1] - 1)
            a, b, c = st.columns(3)
            a.metric("전략 누적 수익률", f"{total:.2%}")
            b.metric("매수 후 보유", f"{bh:.2%}")
            c.metric("포지션 변경 횟수", f"{int(bt['position'].diff().abs().fillna(0).sum())}")
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=bt.index, y=bt["strategy_equity"], name="전략"))
            chart.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="Buy & Hold"))
            chart.update_layout(title="초기 자산 1.0 기준", xaxis_title="", yaxis_title="자산 가치")
            st.plotly_chart(chart, use_container_width=True)
            st.caption("교육용 단순 백테스트입니다. 거래비용·세금·슬리피지·체결 지연을 완전히 반영하지 않습니다. 신호는 다음 거래일에 적용합니다.")

with tabs[3]:
    st.subheader("DCF 기업가치 시나리오")
    st.write("재무 수치는 사용자가 입력해야 합니다. 자동으로 실제 기업 공시 수치라고 가정하지 않습니다.")
    col1, col2, col3 = st.columns(3)
    with col1:
        fcf = st.number_input("기준 연도 FCF (단위: 입력한 통화)", value=1_000_000_000.0, step=100_000_000.0, format="%.0f")
        growth = st.number_input("명시적 기간 연간 성장률 (%)", min_value=-90.0, max_value=200.0, value=8.0) / 100
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
        st.caption("DCF는 입력 가정에 매우 민감합니다. 이 결과는 투자 추천이나 확정 가치가 아닙니다.")

with tabs[4]:
    st.subheader("Ollama 금융 AI 애널리스트")
    st.write("기업 분석, 위험 요인 정리, 전략 설명, 재무 데이터 해석을 자연어로 요청할 수 있습니다.")
    if not ollama_is_configured():
        st.warning("Ollama API 설정이 없습니다. 저장소의 README에 있는 Streamlit Secrets 설정 방법을 따라주세요.")
    prompt = st.text_area(
        "분석 요청",
        value=f"{', '.join(available)}의 최근 시장 데이터를 바탕으로 분석 프레임워크를 제시해줘. 확인된 수치와 추정/가정을 구분하고, 위험 요인과 추가로 확인해야 할 자료를 알려줘.",
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
        "portfolio_metrics": metrics if "metrics" in locals() else None,
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
                answer = ollama_chat(prompt, context)
                st.markdown(answer)
            except Exception as exc:
                st.error(f"AI 요청 실패: {exc}")
    st.caption("LLM은 오류·환각을 일으킬 수 있습니다. 숫자 계산은 가능한 한 정량 엔진의 결과를 기준으로 검토하세요.")

with tabs[5]:
    st.subheader("가격 데이터")
    st.dataframe(prices.tail(30).sort_index(ascending=False), use_container_width=True)
    st.download_button(
        "가격 데이터 CSV 다운로드",
        data=prices.to_csv(index_label="Date").encode("utf-8-sig"),
        file_name="market_prices.csv",
        mime="text/csv"
    )
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
    '<div class="small-note">Aegis는 교육·연구용 오픈소스 기반 분석 도구입니다. '
    '기관용 Aladdin의 전체 기능, 데이터 계약, 운영 통제, 검증 체계를 대체하지 않습니다. '
    '투자 결정 전 원천 데이터와 계산 결과를 독립적으로 확인하세요.</div>',
    unsafe_allow_html=True
)
