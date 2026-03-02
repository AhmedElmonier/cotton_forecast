import sys
sys.path.insert(0, 'd:/Gold')

import logging
logging.basicConfig(level=logging.INFO)

from src.data_fetcher import fetch_all_data, preprocess_data
from src.model import GoldForecastModel, generate_insights
from src.sentiment import analyze_gold_headlines
from src.alerter import format_alert_message
from src.charting import generate_forecast_chart

def main():
    print("Testing Gold bot...")
    df = fetch_all_data("1y")
    if df.empty:
        print("Empty dataframe.")
        return
    
    proc_df = preprocess_data(df)
    model = GoldForecastModel()
    model.fit(proc_df)
    forecast = model.predict(proc_df, days_ahead=5)
    insights = generate_insights(forecast, proc_df, days_ahead=5)
    sentiment = analyze_gold_headlines()
    insights['sentiment_label'] = sentiment['label']
    insights['sentiment_score'] = sentiment['score']
    insights['sentiment_count'] = sentiment['article_count']
    msg = format_alert_message(insights)
    print("Got message:", msg)
    
if __name__ == '__main__':
    main()
