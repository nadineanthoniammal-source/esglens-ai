"""ML component: TF-IDF + Logistic Regression that tags paragraphs as E / S / G / Other.

NOTE: trained on a small seed set. For a stronger result, hand-label 150-300 chunks
from real reports into data/labeled.csv (columns: text,label) - it is auto-loaded.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

SEED = [
    # Environmental
    ("Scope 1 and Scope 2 greenhouse gas emissions decreased by 12 percent compared to last year", "E"),
    ("Total energy consumption from renewable sources increased to 35 percent of our electricity mix", "E"),
    ("We withdrew 2.4 million kilolitres of water and recycled 18 percent of wastewater", "E"),
    ("Hazardous and non-hazardous waste generated was sent for recycling and safe disposal", "E"),
    ("Our net zero target is 2040 with science based targets validated for carbon emissions", "E"),
    ("Installed solar rooftop capacity and purchased renewable energy certificates to cut carbon footprint", "E"),
    ("Air emissions including NOx SOx and particulate matter were within permissible limits", "E"),
    ("Biodiversity conservation and afforestation drives were conducted near our manufacturing plants", "E"),
    ("Energy intensity per unit of production reduced through efficiency and electrification of fleet", "E"),
    ("Scope 3 value chain emissions include purchased goods, business travel and employee commuting", "E"),
    ("Plastic waste management and extended producer responsibility targets were achieved", "E"),
    ("Climate risk assessment and decarbonisation roadmap were approved by the sustainability committee", "E"),
    # Social
    ("Women constitute 28 percent of our workforce and 22 percent of leadership roles", "S"),
    ("Lost time injury frequency rate improved and zero fatalities were recorded this year", "S"),
    ("We invested in employee training with an average of 42 hours of learning per employee", "S"),
    ("Corporate social responsibility spend supported education and healthcare in rural communities", "S"),
    ("Employee turnover rate and diversity and inclusion initiatives for persons with disabilities", "S"),
    ("Health and safety audits of contract workers and occupational health camps were conducted", "S"),
    ("Human rights due diligence and grievance redressal mechanism for workers and communities", "S"),
    ("Customer data privacy complaints and product safety recalls were reported to stakeholders", "S"),
    ("Minimum wages and benefits were provided to all permanent and contractual employees", "S"),
    ("Skill development and livelihood programmes benefited over fifty thousand beneficiaries", "S"),
    ("Supplier code of conduct covers child labour and forced labour prohibition", "S"),
    ("Employee engagement survey and parental leave policy returned 95 percent retention", "S"),
    # Governance
    ("The board of directors comprises 50 percent independent directors with an audit committee", "G"),
    ("Our code of conduct and anti-corruption and anti-bribery policy applies to all employees", "G"),
    ("Whistleblower policy and ethics helpline received complaints which were investigated", "G"),
    ("Remuneration of directors and key managerial personnel was approved by the nomination committee", "G"),
    ("Risk management framework and internal financial controls are reviewed by the board annually", "G"),
    ("Related party transactions were disclosed and approved as per regulatory requirements", "G"),
    ("Shareholder rights, disclosures and compliance with SEBI listing regulations were ensured", "G"),
    ("Cybersecurity and data protection governance is overseen by the risk management committee", "G"),
    ("Statutory auditors and secretarial auditors submitted reports on corporate governance", "G"),
    ("Conflict of interest and fair competition policies were communicated to the board", "G"),
    # Other
    ("The registered office of the company is located at the address mentioned below", "Other"),
    ("Revenue from operations grew due to strong demand in our core markets this quarter", "Other"),
    ("This report has been prepared in accordance with the format prescribed and should be read with notes", "Other"),
    ("Contact the investor relations team for further details about the annual general meeting", "Other"),
    ("The company was incorporated and its equity shares are listed on the stock exchanges", "Other"),
    ("Page intentionally left blank table of contents index and abbreviations used in this document", "Other"),
]


def _training_frame() -> pd.DataFrame:
    df = pd.DataFrame(SEED, columns=["text", "label"])
    extra = Path(__file__).resolve().parent.parent / "data" / "labeled.csv"
    if extra.exists():
        df = pd.concat([df, pd.read_csv(extra)[["text", "label"]]], ignore_index=True)
    return df


def build_model():
    df = _training_frame()
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=5.0),
    )
    model.fit(df["text"], df["label"])
    return model


def tag_chunks(chunks: list[dict]) -> pd.DataFrame:
    model = build_model()
    texts = [c["text"] for c in chunks]
    preds = model.predict(texts)
    conf = model.predict_proba(texts).max(axis=1)
    return pd.DataFrame({"page": [c["page"] for c in chunks], "label": preds,
                         "confidence": conf.round(2), "text": texts})


if __name__ == "__main__":
    df = _training_frame()
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
                          LogisticRegression(max_iter=1000, C=5.0))
    scores = cross_val_score(model, df["text"], df["label"], cv=3, scoring="f1_macro")
    print(f"Samples: {len(df)} | 3-fold macro-F1: {scores.mean():.2f} (+/- {scores.std():.2f})")
