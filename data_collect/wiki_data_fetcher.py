# data_collect/wiki_data_fetcher.py
import os
import json
import yfinance as yf

try:
    import wikipedia
except ImportError:  # wikipedia 패키지가 없으면 메시지 출력 후 패스
    wikipedia = None


class WikiDataCollector:
    """
    티커 리스트의 회사명을 확인한 뒤, 위키백과 요약(2문장)을 수집하여 저장
    ./data/wiki_texts.json : { "AAPL": "텍스트...", ... }
    """

    def __init__(self, tickers):
        self.tickers = tickers
        self.out_file = "./data/wiki_texts.json"
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)

    def fetch(self):
        if wikipedia is None:
            print(
                "[Error] 'wikipedia' 패키지가 설치되지 않았습니다. "
                "pip install wikipedia 로 설치 후 다시 실행하세요."
            )
            return

        result = {}
        for ticker in self.tickers:
            # 회사 영문 풀네임 확인
            try:
                t = yf.Ticker(ticker)
                company = (
                    t.info.get("longName")
                    or t.info.get("shortName")
                    or ticker
                )
            except Exception as e:
                print(f"[Warn] Unable to get company name for {ticker}: {e}")
                company = ticker

            # 위키백과 요약 수집
            try:
                print(f"[Info] Fetching wiki for {company} ({ticker})...")
                summary = wikipedia.summary(
                    company, sentences=2, auto_suggest=False
                )
                result[ticker] = summary
            except wikipedia.exceptions.DisambiguationError as e:
                # 동음이의어 처리 – 첫 번째 항목 선택
                try:
                    summary = wikipedia.summary(
                        e.options[0], sentences=2, auto_suggest=False
                    )
                    result[ticker] = summary
                    print(f"[Warn] {company} 모호, '{e.options[0]}' 사용")
                except Exception as e2:
                    print(f"[Error] Wiki fail for {company}: {e2}")
                    result[ticker] = ""
            except wikipedia.exceptions.PageError:
                print(f"[Warn] No wiki page for {company}.")
                result[ticker] = ""
            except Exception as e:
                print(f"[Error] Wikipedia fetch failed for {company}: {e}")
                result[ticker] = ""

        # 저장: 누락된 항목에는 빈 문자열로 저장하여 모든 티커 키가 존재하도록 함
        with open(self.out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {self.out_file}")
