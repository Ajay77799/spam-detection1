# SpamShield AI – Email Spam Detection using Naive Bayes

A web-based email spam detector built on Bayes' theorem. Each email is cleaned, converted to TF-IDF features and classified by a Multinomial Naive Bayes model that returns the probability of spam.

## Features
- Live prediction with a spam-probability gauge
- Word evidence: which words pushed the decision towards spam or legitimate
- Link safety check (URL shorteners, odd domains, IP addresses, http links)
- Model dashboard: confusion matrix, ROC curve, top words, word clouds
- Batch scan of CSV files with downloadable results
- Train on the built-in dataset, an uploaded CSV, or a CSV loaded from a URL

## Run locally
```bash
git clone https://github.com/<your-username>/spam-detection.git
cd spam-detection
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501. Use `streamlit run`, not `python app.py`.

## Dataset
The app generates a built-in dataset of 1,000 emails (971 unique), so it runs without any extra file. You can upload your own CSV (columns: `label` = spam/ham, `text`) from the sidebar.

## Results (built-in generated dataset, 195 unseen test emails)
| Metric | Value |
|---|---|
| Accuracy | 99.5% |
| Precision (spam) | 100% |
| Recall (spam) | 99.0% |

These scores are on generated emails and are optimistic. A model trained on a less varied dataset scored only 65.0% on the main dataset, so real, varied mail is needed for real-world accuracy.

## Team
Mathankumar M (RA2411056050048), Prakash U (RA2411056050023), Naranthiran (RA2411056050027), Aseel (RA2411056050053)

Guide: Dr. R. Baskar, Assistant Professor, School of Computing, SRM Institute of Science and Technology, Tiruchirappalli
