import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support, classification_report, hamming_loss, jaccard_score
import requests
from bs4 import BeautifulSoup
import joblib
import os
from urllib.parse import urlparse
import html

label_columns = ['Software company', 'Marketing agency', 'Legal services', 'Advertising agency',
                 'Restaurant', 'Solar energy company', 'Travel agency', 'E-commerce service',
                 'Real estate agency', 'Life insurance agency', 'College']

# Strong domain keyword signals for category prediction when scraping is limited
DOMAIN_SIGNALS = {
    'E-commerce service': ['shop', 'store', 'buy', 'cart', 'checkout', 'amazon', 'ebay', 'etsy', 'shopify', 'walmart', 'target', 'flipkart', 'alibaba', 'market', 'mall', 'deal', 'sale', 'product', 'order', 'commerce'],
    'Restaurant': ['food', 'restaurant', 'cafe', 'kitchen', 'grill', 'pizza', 'burger', 'dine', 'menu', 'chef', 'eat', 'bistro', 'diner', 'mcdonalds', 'starbucks', 'subway', 'taco', 'sushi', 'bakery', 'catering'],
    'Software company': ['tech', 'software', 'app', 'digital', 'solution', 'code', 'dev', 'microsoft', 'google', 'apple', 'oracle', 'sap', 'ibm', 'intel', 'saas', 'cloud', 'api', 'data', 'artificial', 'machine', 'automation'],
    'Travel agency': ['travel', 'trip', 'tour', 'hotel', 'flight', 'vacation', 'booking', 'expedia', 'airbnb', 'tripadvisor', 'holiday', 'getaway', 'cruise', 'destination', 'resort', 'carriage'],
    'Real estate agency': ['realty', 'property', 'estate', 'home', 'house', 'apartment', 'rent', 'mortgage', 'realtor', 'zillow', 'redfin', 'condo', 'land', 'housing', 'sale', 'broker'],
    'Legal services': ['law', 'legal', 'attorney', 'lawyer', 'counsel', 'litigation', 'solicitor', 'justice', 'firm', 'court', 'barrister', 'paralegal'],
    'Marketing agency': ['marketing', 'seo', 'brand', 'advertise', 'media', 'growth', 'campaign', 'promo', 'ppc', 'digital', 'social', 'content', 'creative', 'strategy', 'agency', 'public'],
    'Advertising agency': ['ad', 'advert', 'creative', 'agency', 'media', 'campaign', 'billboard', 'display', 'brand', 'promotion', 'marketing', 'publicity', 'copywriting', 'production'],
    'Solar energy company': ['solar', 'energy', 'renewable', 'panel', 'green', 'power', 'sun', 'electric', 'photovoltaic', 'battery', 'storage', 'efficiency', 'sustainable'],
    'Life insurance agency': ['insurance', 'life', 'policy', 'coverage', 'premium', 'protect', 'mutual', 'statefarm', 'allstate', 'geico', 'financial', 'agent', 'broker'],
    'College': ['college', 'university', 'edu', 'campus', 'degree', 'academy', 'school', 'harvard', 'mit', 'stanford', 'oxford', 'learn', 'education', 'student', 'admission', 'faculty']
}

# Category priors based on training data (updated after training)
CATEGORY_PRIORS = {
    'Software company': 0.185,
    'Marketing agency': 0.189,
    'Legal services': 0.092,
    'Advertising agency': 0.150,
    'Restaurant': 0.091,
    'Solar energy company': 0.091,
    'Travel agency': 0.092,
    'E-commerce service': 0.112,
    'Real estate agency': 0.095,
    'Life insurance agency': 0.092,
    'College': 0.091
}

# Keywords that indicate bot protection / blocked scraping
BOT_BLOCK_INDICATORS = [
    'javascript is disabled',
    'enable javascript',
    "verify that you're not a robot",
    'verify you are not a robot',
    'captcha',
    'robot check',
    'access denied',
    'blocked',
    'please enable cookies',
    'security check',
    'human verification',
    'automated access',
    'bot detection',
    'cloudflare',
    'ddos protection',
    'please wait',
    'redirecting',
    'checking your browser',
    'enable js',
    'turn on javascript'
]


def clean_text(text):
    """Clean text while preserving some meaningful signals."""
    if not isinstance(text, str):
        text = str(text)
    # Decode HTML entities
    text = html.unescape(text)
    # Lowercase
    text = text.lower()
    # Keep dots, hyphens within words (e.g., e-commerce, co.uk)
    text = re.sub(r'[^a-z0-9\s\.\-]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_bot_blocked(text):
    """Check if the scraped text indicates bot protection."""
    text_lower = text.lower()
    for indicator in BOT_BLOCK_INDICATORS:
        if indicator in text_lower:
            return True
    return False


def get_domain_scores(url):
    """Get category scores (0-1) based on URL domain keywords."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    domain = re.sub(r'^www\.', '', domain)
    domain_name = domain.split('.')[0] if domain.split('.') else ''
    path = parsed.path.lower()

    scores = {}
    for category, keywords in DOMAIN_SIGNALS.items():
        score = 0.0
        for kw in keywords:
            if kw == domain_name:
                score = max(score, 0.90)  # Exact domain match = very high confidence
            elif kw in domain_name:
                score = max(score, 0.55)  # Partial domain match
            elif kw in domain:
                score = max(score, 0.35)  # Anywhere in domain
            elif kw in path:
                score = max(score, 0.25)  # In URL path
        if score > 0:
            scores[category] = score
    return scores


def predict_from_domain_only(url):
    """Fallback prediction using domain heuristics + category priors when scraping fails."""
    scores = get_domain_scores(url)

    results = []
    for label in label_columns:
        # Combine domain score with category prior
        domain_score = scores.get(label, 0.0)
        prior = CATEGORY_PRIORS.get(label, 0.09)
        # Weighted combination: domain signals get 70% weight, priors 30%
        prob = (domain_score * 0.70) + (prior * 0.30)
        prob = min(prob * 100, 95.0)
        match = ''
        results.append({'Category': label, 'Probability (%)': f"{prob:.1f}%", 'Match': match})

    results.sort(key=lambda x: float(x['Probability (%)'][:-1]), reverse=True)
    note = "Domain-only prediction (website blocked automated access)"
    return results, note


def scrape_website(url):
    """Scrape website content with robust headers and bot detection."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }

    try:
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()

        if len(response.text.strip()) < 100:
            raise Exception(f"BLOCKED:{response.status_code}")

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove non-content elements
        for elem in soup(["script", "style", "nav", "footer", "header", "aside"]):
            elem.extract()

        text = soup.get_text(separator=' ', strip=True)
        title = soup.title.string if soup.title else ''
        h1s = ' '.join([tag.get_text() for tag in soup.find_all('h1')])
        h2s = ' '.join([tag.get_text() for tag in soup.find_all('h2')])
        h3s = ' '.join([tag.get_text() for tag in soup.find_all('h3')])
        ps = ' '.join([tag.get_text() for tag in soup.find_all('p')])

        # Extract meta descriptions
        meta_desc = ''
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta and meta.get('content'):
            meta_desc = meta['content']

        # Extract Open Graph tags
        og_title = ''
        og = soup.find('meta', attrs={'property': 'og:title'})
        if og and og.get('content'):
            og_title = og['content']

        og_desc = ''
        og = soup.find('meta', attrs={'property': 'og:description'})
        if og and og.get('content'):
            og_desc = og['content']

        # Extract meta keywords
        meta_keywords = ''
        meta_kw = soup.find('meta', attrs={'name': 'keywords'})
        if meta_kw and meta_kw.get('content'):
            meta_keywords = meta_kw['content']

        # Extract twitter cards
        twitter_title = ''
        tw = soup.find('meta', attrs={'name': 'twitter:title'})
        if tw and tw.get('content'):
            twitter_title = tw['content']

        twitter_desc = ''
        tw = soup.find('meta', attrs={'name': 'twitter:description'})
        if tw and tw.get('content'):
            twitter_desc = tw['content']

        # Extract anchor text from navigation
        nav_links = soup.find_all('a', href=True)
        nav_text = ' '.join([a.get_text() for a in nav_links[:30]])  # Limit to first 30 links

        # Weighted combination: title and meta get higher weight by duplication
        combined = (
            f"{title} {title} "
            f"{meta_desc} {meta_desc} "
            f"{og_title} {og_title} "
            f"{og_desc} {og_desc} "
            f"{twitter_title} {twitter_desc} "
            f"{meta_keywords} "
            f"{h1s} {h1s} "
            f"{h2s} {h2s} "
            f"{h3s} "
            f"{ps} "
            f"{nav_text} "
            f"{text}"
        )
        cleaned = clean_text(combined)

        word_count = len(cleaned.split())

        if word_count < 10:
            raise Exception(f"BLOCKED:{response.status_code}")

        if is_bot_blocked(cleaned):
            raise Exception("BLOCKED:BOT")

        return cleaned
    except requests.exceptions.Timeout:
        raise Exception(f"Timeout scraping {url}. The site may be slow or blocking automated requests.")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Could not connect to {url}. Please check the URL and try again.")
    except Exception as e:
        err_str = str(e)
        if err_str.startswith("BLOCKED:"):
            raise
        raise Exception(f"Error scraping {url}: {err_str}")


def extract_url_features(url):
    """Extract text features from URL for the vectorizer."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    domain = re.sub(r'^www\.', '', domain)
    path = parsed.path.lower().replace('/', ' ').replace('-', ' ').replace('_', ' ')

    # Combine domain parts and path as extra text
    domain_parts = domain.replace('.', ' ')
    return f"{domain_parts} {path}"


# Global model & vectorizer cache
_model = None
_vectorizer = None


def load_model():
    """Load trained model and vectorizer from disk (cached)."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
            raise FileNotFoundError("Model files not found. Run train_model.py first.")
        _model = joblib.load('model.pkl')
        _vectorizer = joblib.load('vectorizer.pkl')
    return _model, _vectorizer


def predict_website(url):
    """Predict categories for a single URL.
    
    Returns a dict with:
        status: 'success' | 'error'
        url: normalized URL
        predictions: list of {Category, Probability (%), Match}
        note: optional note string
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    model, vectorizer = load_model()

    try:
        scraped_text = scrape_website(url)
        url_features = extract_url_features(url)
        combined = f"{scraped_text} {url_features}"
        X = vectorizer.transform([combined])
        probs = model.predict_proba(X)[0]
    except Exception as e:
        err_str = str(e)
        if err_str.startswith("BLOCKED:") or "Timeout" in err_str or "Could not connect" in err_str:
            predictions, note = predict_from_domain_only(url)
            return {
                'status': 'success',
                'url': url,
                'predictions': predictions,
                'note': note
            }
        raise

    predictions = []
    for idx, label in enumerate(label_columns):
        prob = probs[idx] * 100
        match = '✅' if prob >= 50 else ''
        predictions.append({
            'Category': label,
            'Probability (%)': f"{prob:.1f}%",
            'Match': match
        })

    predictions.sort(key=lambda x: float(x['Probability (%)'][:-1]), reverse=True)

    return {
        'status': 'success',
        'url': url,
        'predictions': predictions
    }


def predict_batch(urls):
    """Predict categories for multiple URLs.
    
    Returns a list of result dicts (same shape as predict_website output).
    """
    results = []
    for url in urls:
        try:
            result = predict_website(url)
            results.append(result)
        except Exception as e:
            results.append({
                'status': 'error',
                'url': url,
                'error': str(e)
            })
    return results


def train_model(csv_path='sample_11_categories_1.csv'):
    """Train and save model + vectorizer with evaluation."""
    df = pd.read_csv(csv_path)

    # Build combined text with structure awareness
    df['combined_text'] = df[['text', 'html_title', 'h1', 'h2', 'p']].fillna('').agg(' '.join, axis=1)
    df['clean_text'] = df['combined_text'].apply(clean_text)

    X = df['clean_text']
    y = df[label_columns]

    # Stratified-like split for multi-label using iterative stratification is ideal,
    # but simple random split is acceptable for this dataset size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Improved TF-IDF vectorizer: more features, char ngrams, sublinear tf
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        analyzer='word',
        token_pattern=r'(?u)\b\w\w+\b'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training OneVsRestClassifier with LogisticRegression...")
    # Use OneVsRestClassifier with LogisticRegression - the gold standard for text classification
    base_clf = LogisticRegression(
        random_state=42,
        max_iter=2000,
        C=1.0,
        class_weight='balanced',
        solver='lbfgs',
        n_jobs=-1
    )
    ovr = OneVsRestClassifier(base_clf, n_jobs=-1)
    ovr.fit(X_train_tfidf, y_train)

    print("Calibrating probabilities for reliable confidence scores...")
    # Calibrate probabilities for reliable confidence scores
    calibrated = OneVsRestClassifier(
        CalibratedClassifierCV(
            LogisticRegression(random_state=42, max_iter=2000, C=1.0, class_weight='balanced', n_jobs=-1),
            method='sigmoid',
            cv=3
        ),
        n_jobs=-1
    )
    calibrated.fit(X_train_tfidf, y_train)

    # Evaluation on test set
    y_pred = ovr.predict(X_test_tfidf)
    y_pred_cal = calibrated.predict(X_test_tfidf)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION REPORT")
    print("=" * 70)

    print("\n--- Base OneVsRest LogisticRegression ---")
    print(classification_report(y_test, y_pred, target_names=label_columns, zero_division=0))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='micro', zero_division=0
    )
    print(f"Micro Avg — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    print(f"Macro Avg — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
    try:
        jac = jaccard_score(y_test, y_pred, average='micro', zero_division=0)
        print(f"Jaccard Score (micro): {jac:.4f}")
    except Exception:
        pass

    print("\n--- Calibrated OneVsRest LogisticRegression ---")
    print(classification_report(y_test, y_pred_cal, target_names=label_columns, zero_division=0))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_cal, average='micro', zero_division=0
    )
    print(f"Micro Avg — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_cal, average='macro', zero_division=0
    )
    print(f"Macro Avg — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    print(f"Hamming Loss: {hamming_loss(y_test, y_pred_cal):.4f}")
    try:
        jac = jaccard_score(y_test, y_pred_cal, average='micro', zero_division=0)
        print(f"Jaccard Score (micro): {jac:.4f}")
    except Exception:
        pass

    print("\n" + "=" * 70)

    # Save model and vectorizer
    joblib.dump(calibrated, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Saved model.pkl and vectorizer.pkl")

    # Update priors based on training data
    priors = y.mean().to_dict()
    print("\nUpdated CATEGORY_PRIORS based on training data:")
    for label in label_columns:
        print(f"  '{label}': {priors.get(label, 0.09):.3f},")

    return calibrated, vectorizer


if __name__ == '__main__':
    train_model()

