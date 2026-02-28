import sys
sys.path.append('d:/Cotton')
from src.data_fetcher import fetch_all_data, preprocess_data
from src.model import CottonForecastModel, generate_insights
from src.sentiment import analyze_cotton_headlines
from src.alerter import format_alert_message
from src.charting import generate_forecast_chart

def main():
    print("Fetching data...")
    raw_df = fetch_all_data(period="1y")
    if raw_df.empty:
        print("Data is empty!")
        return
    process_df = preprocess_data(raw_df)
    print("Data processed. Fitting model...")
    model = CottonForecastModel()
    model.fit(process_df)
    print("Predicting...")
    forecast_df = model.predict(process_df, days_ahead=30)
    print("Generating insights...")
    insights = generate_insights(forecast_df, process_df, days_ahead=30)
    print("Analyzing sentiment...")
    sentiment = analyze_cotton_headlines()
    insights['sentiment_label'] = sentiment['label']
    insights['sentiment_score'] = sentiment['score']
    insights['sentiment_count'] = sentiment['article_count']
    print("Formatting message...")
    msg = format_alert_message(insights)
    print("Generating chart...")
    chart = generate_forecast_chart(process_df, forecast_df)
    print("Success!")

if __name__ == '__main__':
    main()
