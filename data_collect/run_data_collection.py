"""
S&P500 티커 리스트 기반 데이터 일괄 수집 드라이버
사용 모듈:
 - StockDataCollector
 - WikiDataCollector
 - MacroDataCollector
"""
import json

from .stock_data_fetcher import StockDataCollector
from .wiki_data_fetcher import WikiDataCollector
from .macro_data_fetcher import MacroDataCollector


def _load_tickers():
    """
    config/SnP500_list.json 파일에서 'symbol' 값만 추출
    """
    with open("config/SnP500_list.json", encoding="utf-8") as f:
        data = json.load(f)
    return [c["symbol"] for c in data["companies"]]


def run():
    tickers = _load_tickers()
    print(f"[Info] {len(tickers)} tickers loaded from SnP500_list.json")

    # 1) 주가 & 재무제표
    stock_collector = StockDataCollector(tickers)
    stock_collector.fetch_prices()
    stock_collector.fetch_financials()

    # 2) 위키 정보
    wiki_collector = WikiDataCollector(tickers)
    wiki_collector.fetch()

    # 3) 거시경제 지표
    macro_collector = MacroDataCollector()
    macro_collector.fetch()


# CLI 직접 실행 시
if __name__ == "__main__":
    run()
