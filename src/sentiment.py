import yfinance as yf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import os
import json

try:
    import google.generativeai as genai
except ImportError:
    pass # Will be handled later if missing

logger = logging.getLogger(__name__)

# Ensure the VADER lexicon is downloaded (only downloads if not present)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading NLTK VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)

def analyze_cotton_headlines() -> dict:
    """
    Fetches the latest news headlines for Cotton from Yahoo Finance
    and calculates an average sentiment score using Gemini AI (with VADER fallback).
    
    Returns:
        dict: Containing 'score' (-1 to 1), 'label' (String), 'article_count', and 'summary'.
    """
    logger.info("Fetching latest news headlines for Cotton (CT=F)...")
    try:
        cotton = yf.Ticker("CT=F")
        news = cotton.news
        
        if not news:
            logger.warning("No recent news found for Cotton.")
            return {"score": 0.0, "label": "No Data", "article_count": 0, "summary": "No recent news available."}
            
        headlines = [article.get('title', '') for article in news if article.get('title', '')]
        valid_articles = len(headlines)
        
        if valid_articles == 0:
            return {"score": 0.0, "label": "No Data", "article_count": 0, "summary": "No recent news available."}

        # Try Gemini AI first
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                prompt = (
                    "You are a financial market analyst specializing in agricultural commodities, particularly cotton.\n"
                    "Analyze the following recent news headlines about cotton and provide a sentiment analysis.\n"
                    "Respond ONLY with a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "score": <float between -1.0 (extremely bearish) and 1.0 (extremely bullish)>,\n'
                    '  "label": <string, exactly one of "Optimistic 🟢", "Pessimistic 🔴", or "Neutral 🟡">,\n'
                    '  "summary": <string, a concise 1-2 sentence summary of the market context based on the headlines>\n'
                    "}\n\n"
                    "Headlines:\n"
                )
                for h in headlines:
                    prompt += f"- {h}\n"
                
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                    
                ai_data = json.loads(response_text)
                logger.info(f"AI Sentiment Analysis successful: {ai_data['label']}")
                return {
                    "score": float(ai_data.get("score", 0.0)),
                    "label": ai_data.get("label", "Neutral 🟡"),
                    "article_count": valid_articles,
                    "summary": ai_data.get("summary", "AI summarized the news successfully.")
                }
            except Exception as ai_e:
                logger.warning(f"Gemini AI analysis failed, falling back to VADER: {ai_e}")
                
        # Fallback to VADER
        logger.info("Using VADER for sentiment analysis (fallback).")
        sia = SentimentIntensityAnalyzer()
        total_score = sum(sia.polarity_scores(title)['compound'] for title in headlines)
        avg_score = total_score / valid_articles
        
        if avg_score > 0.2:
            label = "Optimistic 🟢"
        elif avg_score < -0.2:
            label = "Pessimistic 🔴"
        else:
            label = "Neutral 🟡"
            
        logger.info(f"VADER Analyzed {valid_articles} headlines. Average Sentiment: {avg_score:.2f} ({label})")
        
        return {
            "score": avg_score,
            "label": label,
            "article_count": valid_articles,
            "summary": "Basic sentiment analysis performed. AI enhancements require an API key."
        }
        
    except Exception as e:
        logger.error(f"Error fetching/analyzing news sentiment: {e}")
        return {"score": 0.0, "label": "Error", "article_count": 0, "summary": "Error analyzing news."}

if __name__ == "__main__":
    result = analyze_cotton_headlines()
    print(f"Sentiment Result: {result}")
