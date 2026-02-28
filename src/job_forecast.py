import sys
import os
import logging
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add parent directory to path so imports work correctly when run from GitHub Actions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import fetch_all_data, preprocess_data
from src.model import CottonForecastModel, generate_insights
from src.sentiment import analyze_cotton_headlines
from src.alerter import format_alert_message, send_telegram_alert
from src.charting import generate_forecast_chart

def run_job():
    logger.info("Starting scheduled Cotton Forecast job...")
    
    # Use 5y period for more accurate Prophet modeling, matching bot.py
    raw_df = fetch_all_data(period="5y")
    if raw_df.empty:
        logger.error("Data is empty. Cannot proceed.")
        sys.exit(1)
        
    process_df = preprocess_data(raw_df)
    
    model = CottonForecastModel()
    model.fit(process_df)
    
    forecast_df = model.predict(process_df, days_ahead=30)
    insights = generate_insights(forecast_df, process_df, days_ahead=30)
    
    sentiment = analyze_cotton_headlines()
    insights['sentiment_label'] = sentiment['label']
    insights['sentiment_score'] = sentiment['score']
    insights['sentiment_count'] = sentiment['article_count']
    
    msg = format_alert_message(insights)
    chart = generate_forecast_chart(process_df, forecast_df)
    
    logger.info("Sending alert to Telegram...")
    success = send_telegram_alert(msg, image_path=chart)
    
    if success:
        logger.info("Scheduled job completed successfully.")
    else:
        logger.error("Failed to send Telegram alert. Ensure secrets are configured correctly.")
        sys.exit(1)

if __name__ == '__main__':
    run_job()
