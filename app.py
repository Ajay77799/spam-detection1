"""SpamShield AI - interactive Naive Bayes email spam detector (Streamlit).
Run:  streamlit run app.py"""
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"]{font-family:'Inter',sans-serif}
.hero{background:linear-gradient(120deg,#4f46e5,#9333ea 60%,#ec4899);padding:28px 32px;border-radius:18px;color:#fff;margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem;font-weight:800;color:#fff;padding:0}
.hero p{margin:6px 0 0;opacity:.92}
.pill{display:inline-block;background:rgba(255,255,255,.2);padding:4px 12px;border-radius:99px;margin:12px 8px 0 0;font-size:.85rem}
.verdict{border-radius:16px;padding:22px;text-align:center;animation:pop .4s ease}
.verdict.spam{background:rgba(239,68,68,.13);border:2px solid #ef4444}
.verdict.ham{background:rgba(34,197,94,.13);border:2px solid #22c55e}
.verdict h2{margin:0;font-size:1.8rem;padding:0}.verdict p{margin:6px 0 0;opacity:.8}
@keyframes pop{from{transform:scale(.94);opacity:0}to{transform:scale(1);opacity:1}}
.chip{display:inline-block;padding:3px 11px;border-radius:99px;margin:3px 4px 3px 0;font-size:.85rem;font-weight:600}
.chip.s{background:rgba(239,68,68,.15);color:#ef4444}.chip.h{background:rgba(34,197,94,.15);color:#16a34a}
.flag{padding:8px 12px;border-radius:10px;margin:4px 0;background:rgba(245,158,11,.14);font-size:.9rem}
</style>""", unsafe_allow_html=True)

SAMPLES = {
    "🎁 Prize scam": "URGENT: Congratulations! You have won a $5000 gift card. Click here http://bit.ly/claim99 within 24 hours or lose your prize!!!",
    "🏦 Phishing": "Security alert: your bank account has been suspended. Verify your password immediately at http://secure-login.xyz/verify to avoid permanent closure.",
    "📅 Work mail": "Hi team, the meeting is rescheduled to Monday at 10 am. Please find attached the report for the DBMS project. Thanks and regards.",
    "🎓 College notice": "Dear students, the assignment submission is extended to Friday. Please bring your lab record. All the best.",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = text.replace("$", " dollarsign ").replace("!", " exclam ")
    text = re.sub(r"\d+", " numtoken ", text)
    return re.sub(r"[^a-z' ]", " ", text)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    lc = next((cols[k] for k in ("label", "v1", "category", "class") if k in cols), None)
    tc = next((cols[k] for k in ("text", "v2", "message", "body", "email") if k in cols), None)
    if lc is None or tc is None:
        raise ValueError(f"Need a label column and a text column. Found: {list(df.columns)}")
    d = df[[lc, tc]].dropna()
    d.columns = ["label", "text"]
    d["label"] = d.label.astype(str).str.lower().str.strip().map(
        {"spam": "spam", "1": "spam", "1.0": "spam", "ham": "ham", "0": "ham", "0.0": "ham", "not spam": "ham"})
    d = d.dropna().drop_duplicates("text").reset_index(drop=True)
    d["y"] = (d.label == "spam").astype(int)
    if d.y.nunique() < 2 or len(d) < 40:
        raise ValueError("Dataset needs at least 40 rows with both spam and ham labels.")
    return d


def read_csv_any(src):
    for enc in ("utf-8", "latin-1"):
        try:
            if hasattr(src, "seek"):
                src.seek(0)
            return pd.read_csv(src, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file")


def generate_emails():
    """Built-in demo dataset: 1000 emails (500 spam / 500 ham) - no CSV file needed."""
    random.seed(2026)
    N = 1000
    P = random.choice
    def num(a, b): return random.randint(a, b)
    def url(): return P(["http://bit.ly/", "http://secure-", "https://tinyurl.com/", "http://www.", "http://login-"]) + P(["verify", "claim", "offer", "update", "pay", "win"]) + str(num(10, 9999)) + P([".com", ".biz", ".info", ".xyz"])
    def typo(s):
        if random.random() < 0.25 and len(s) > 8:
            i = num(1, len(s) - 3); s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        return s
    def shout(s): return s.upper() if random.random() < 0.2 else s

    # ---------------- SPAM ----------------
    def s_prize():
        return f"{P(['Congratulations','Dear winner','Hello lucky user'])}! You have been selected to receive {P(['a $','$','USD '])}{num(500,50000)} {P(['cash prize','gift card','lottery reward','cash bonus'])}. {P(['Claim now','Reply with your details','Click here'])}: {url()} {P(['Offer expires in 24 hours!!!','Act fast!','Limited winners only.'])}"
    def s_phish():
        return f"{P(['Dear customer','Security alert','Notice'])}: your {P(['bank','PayPal','Apple ID','Netflix','email'])} account has been {P(['suspended','locked','limited','compromised'])}. {P(['Verify your password','Confirm your identity','Update your billing details'])} immediately at {url()} to avoid {P(['permanent closure','loss of access','extra charges'])}."
    def s_pharma():
        return f"{P(['Buy','Order','Get'])} {P(['cheap meds','generic pills','weight loss pills','miracle supplements'])} online - {P(['no prescription needed','save up to 80%','100% guaranteed','discreet shipping'])}. {P(['Order today','Limited stock','Free shipping on all orders'])} {url()}"
    def s_money():
        return f"{P(['Earn','Make','Turn'])} ${num(200,9000)} {P(['per day','per week','from home'])} with {P(['crypto trading','our secret system','forex signals','this investment'])}. {P(['Guaranteed returns','No experience needed','Risk free','Double your money'])}! {P(['Join now','Start today','Invest now'])} {url()}"
    def s_job():
        return f"{P(['Urgent hiring','Work from home','Part time job'])}: earn ${num(15,90)} per hour, {P(['no interview','no experience required','instant payment'])}. Send your {P(['bank details','full name and phone number','ID copy'])} to {P(['hr','jobs','career'])}{num(1,99)}@{P(['gmail','mail-box','quickjobs'])}.{P(['com','biz','net'])} today."
    def s_invoice():
        return f"{P(['Invoice','Payment receipt','Shipping label','Order confirmation'])} #{num(10000,99999)} attached. {P(['Open the attachment','Download the document','Click the link'])} to {P(['view details','release your package','confirm payment'])}: {url()}"
    def s_stealth():  # spam that looks polite
        return f"Hi, {P(['I found your profile','hope you are doing well','quick question'])}. {P(['We have a special offer for you','You qualify for a free upgrade','Congratulations, you are pre-approved'])}. {P(['Visit','Click','See'])} {url()} {P(['to learn more','for details','to claim'])}."
    SPAM = [s_prize, s_phish, s_pharma, s_money, s_job, s_invoice, s_stealth]

    # ---------------- HAM ----------------
    def h_work():
        return f"{P(['Hi team,','Hello all,','Hi,','Dear colleagues,'])} {P(['the meeting is moved to','please review the report before','reminder: the project deadline is','can we sync about the budget on'])} {P(['Monday','Tuesday','Thursday','Friday'])} {P(['at 10 am','at 3 pm','after lunch','morning'])}. {P(['Let me know if that works.','Thanks and regards.','I will share the notes after.','Please confirm your availability.'])}"
    def h_college():
        return f"{P(['Dear students,','Hi all,','Hello,'])} {P(['the assignment submission is extended to','lab session for DBMS will be held on','internal exam timetable is attached for','project review is scheduled on'])} {P(['next week','Wednesday','the 15th','Friday'])}. {P(['Please prepare your slides.','Bring your lab record.','Contact me for doubts.','All the best.'])}"
    def h_personal():
        return f"{P(['Hey,','Hi da,','Hello,'])} {P(['are we still meeting','happy birthday! hope you have a great day','dinner at home this weekend','thanks for the photos from the trip'])}? {P(['Call me when free.','Let me know.','See you soon.','Miss you.'])}"
    def h_trans():
        return f"{P(['Your order','Your package','Your ticket','Your booking'])} {P(['has been shipped','will be delivered tomorrow','is confirmed','is ready for pickup'])}. {P(['Track it in the app.','Order number','Reference number'])} {num(100000,999999)}. {P(['Thank you for shopping with us.','Have a nice trip.','No action is required.'])}"
    def h_news():  # legit promo / newsletter -> shares words with spam
        return f"{P(['This week in tech:','Your monthly newsletter:','Campus update:'])} {P(['free webinar on machine learning','new offer on student plans','discount on course fee for early registration','urgent: library closes early on Friday'])}. {P(['Register here','Read more','Details on the website'])}. {P(['Unsubscribe anytime.','You are receiving this because you subscribed.'])}"
    def h_otp():
        return f"{P(['Your OTP is','Use code','Verification code:'])} {num(100000,999999)}. {P(['Do not share it with anyone.','Valid for 10 minutes.','If you did not request this, ignore this message.'])}"

    # legit mails that deliberately look like spam categories (real-world overlap)
    def h_secure():
        return f"Security alert: new sign-in to your {P(['Google','bank','Apple ID','college portal'])} account from {P(['Chennai','Bangalore','a new device','Windows'])}. {P(['If this was you, no action is needed.','If this was not you, change your password.','You can review activity in your account settings.'])} https://accounts.{P(['google','apple','college'])}.com/security"
    def h_invoice():
        return f"{P(['Invoice','Receipt','Fee receipt','Payment confirmation'])} #{num(10000,99999)} attached. {P(['Amount due','Total paid','Course fee'])}: ${num(10,500)}. {P(['Please pay by the due date.','Thank you for your payment.','Contact accounts office for queries.'])}"
    def h_prize():
        return f"Congratulations! You won {P(['first prize','the best project award','a place in the finals'])} in the {P(['coding contest','paper presentation','hackathon','quiz'])}. {P(['Collect your certificate on Friday.','Details will be announced soon.','Please bring your ID card.'])}"
    def h_job():
        return f"We are hiring {P(['interns','a data analyst','a software trainee'])}. {P(['Apply with your resume before','Interview will be held on','Send your CV to the HR team by'])} {P(['Monday','the 20th','next Friday'])}. {P(['Salary as per company norms.','Work from home is allowed twice a week.','Free lunch provided.'])}"
    HAM = [h_work, h_college, h_personal, h_trans, h_news, h_otp, h_secure, h_invoice, h_prize, h_job]

    def mix_ham(): return f"{P(['Hi,','Hello,'])} {P(['free to talk later?','please check this offer letter draft','urgent: send the report today','click the calendar link to join'])}. {P(['Thanks.','Regards.'])}"
    def mix_spam(): return f"{P(['Hi,','Hello,'])} {P(['are you free today','please check this','see attached'])}. {url()}"

    def ham_extras(t):
        r = random.random()
        if r < 0.30:
            t += " " + P(["https://meet.google.com/abc-defg-hij", f"https://lms.college.edu/course/{num(100,999)}", f"https://www.amazon.in/orders/{num(10000,99999)}", "https://docs.google.com/document/d/1xYz"])
        if random.random() < 0.15:
            t += f" Total amount: ${num(5,400)}."
        return t

    def spam_extras(t):
        if random.random() < 0.30:   # spam with no link at all
            t = re.sub(r"\S*http\S*", P(["reply to this email", "call now", "message us"]), t, flags=re.I)
        return t

    rows = []
    for _ in range(N // 2):
        t = mix_spam() if random.random() < 0.10 else P(SPAM)()
        rows.append(("spam", shout(typo(spam_extras(t)))))
        t = mix_ham() if random.random() < 0.10 else P(HAM)()
        rows.append(("ham", typo(ham_extras(t))))
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["label", "text"])


@st.cache_data
def builtin_df():
    """Use emails.csv next to app.py if it exists, otherwise generate the demo data."""
    p = Path(__file__).parent / "emails.csv"
    return read_csv_any(str(p)) if p.exists() else generate_emails()


@st.cache_data(show_spinner="Downloading dataset…")
def load_url(url):
    return read_csv_any(url)


@st.cache_resource(show_spinner="Training Naive Bayes model…")
def build(_df, key, alpha, ngram):
    Xtr, Xte, ytr, yte = train_test_split(_df.text, _df.y, test_size=0.2, random_state=42, stratify=_df.y)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english", ngram_range=(1, ngram))),
        ("nb", MultinomialNB(alpha=alpha))])
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(yte, p)
    return model, dict(acc=accuracy_score(yte, pred), pre=precision_score(yte, pred, zero_division=0),
                       rec=recall_score(yte, pred, zero_division=0), f1=f1_score(yte, pred, zero_division=0),
                       cm=confusion_matrix(yte, pred), fpr=fpr, tpr=tpr, auc=auc(fpr, tpr),
                       n_train=len(Xtr), n_test=len(Xte))


def evidence(model, text, k=12):
    tf, nb = model.named_steps["tfidf"], model.named_steps["nb"]
    x = tf.transform([text])
    words = tf.get_feature_names_out()
    lr = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    rows = [(words[i], float(lr[i] * x[0, i])) for i in x.nonzero()[1]]
    return sorted(rows, key=lambda r: -abs(r[1]))[:k]


def analyze_links(text):
    flags, urls = [], re.findall(r"https?://[^\s]+|www\.[^\s]+", text)
    for u in urls:
        host = re.sub(r"^https?://", "", u).split("/")[0].lower()
        if host.split(".")[0] in ("bit", "tinyurl", "t", "goo", "ow") or host in ("bit.ly", "tinyurl.com", "t.co", "goo.gl"):
            flags.append((u, "URL shortener hides the real destination"))
        if re.search(r"\.(xyz|biz|info|top|click|tk|ru)(\W|$)", host):
            flags.append((u, "Suspicious domain ending"))
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}", host):
            flags.append((u, "Raw IP address instead of a domain name"))
        if host.count("-") >= 2 or "@" in u:
            flags.append((u, "Unusual characters in the address"))
        if u.lower().startswith("http://"):
            flags.append((u, "Not encrypted (http instead of https)"))
    return urls, flags


def gauge(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 38}},
        title={"text": "Spam probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#6366f1"},
               "steps": [{"range": [0, 35], "color": "rgba(34,197,94,.35)"},
                         {"range": [35, 65], "color": "rgba(245,158,11,.35)"},
                         {"range": [65, 100], "color": "rgba(239,68,68,.35)"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def hbar(labels, vals, colors, title):
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(title=title, height=max(260, 28 * len(labels) + 90), margin=dict(l=10, r=10, t=45, b=10),
                      yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------- sidebar -------------------------------
with st.sidebar:
    st.header("⚙️ Control panel")
    source = st.radio("Dataset source", ["Built-in demo dataset", "Upload CSV", "Load from URL"])
    raw = None
    try:
        if source.startswith("Built-in"):
            raw = builtin_df()
        elif source == "Upload CSV":
            up = st.file_uploader("CSV with label + text columns", type="csv")
            raw = read_csv_any(up) if up else builtin_df()
            if not up:
                st.caption("No file yet — using built-in data.")
        else:
            url = st.text_input("Public CSV URL", placeholder="https://.../spam.csv")
            raw = load_url(url) if url else builtin_df()
            if not url:
                st.caption("No URL yet — using built-in data.")
        df = normalize(raw)
    except Exception as e:
        st.error(f"Dataset problem: {e}\n\nFalling back to built-in data.")
        df = normalize(builtin_df())
    alpha = st.slider("Laplace smoothing (α)", 0.05, 3.0, 1.0, 0.05)
    ngram = st.selectbox("Word features", [1, 2], format_func=lambda n: "Single words" if n == 1 else "Words + word pairs")
    thr = st.slider("Spam threshold", 0.1, 0.9, 0.5, 0.05, help="Raise it to avoid flagging real mail as spam.")
    st.caption(f"📚 {len(df)} emails · {int(df.y.sum())} spam / {int((1 - df.y).sum())} ham")

key = int(pd.util.hash_pandas_object(df.text).sum() % 10**9)
model, M = build(df, key, alpha, ngram)

st.markdown(f"""<div class="hero"><h1>🛡️ SpamShield AI</h1>
<p>Email spam detection powered by Bayes' theorem (Multinomial Naive Bayes)</p>
<span class="pill">📚 {len(df)} emails</span><span class="pill">🎯 Accuracy {M['acc']:.1%}</span>
<span class="pill">📈 AUC {M['auc']:.3f}</span><span class="pill">⚡ Instant predictions</span></div>""", unsafe_allow_html=True)

st.session_state.setdefault("history", [])
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze email", "📊 Model insights", "📁 Batch scan", "🧠 How it works"])

# ------------------------------- tab 1 -------------------------------
with tab1:
    st.write("**Quick samples:**")
    cols = st.columns(len(SAMPLES))
    for c, (name, txt) in zip(cols, SAMPLES.items()):
        c.button(name, width="stretch", on_click=lambda t=txt: st.session_state.update(mail=t))
    text = st.text_area("Paste an email (subject + body)", key="mail", height=150,
                        placeholder="Type or paste an email here…")
    if text.strip():
        p = float(model.predict_proba([text])[0][1])
        is_spam = p >= thr
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""<div class="verdict {'spam' if is_spam else 'ham'}"><h2>{'🚫 SPAM detected' if is_spam else '✅ Looks legitimate'}</h2>
            <p>{'Do not click links or reply.' if is_spam else 'No strong spam signals found.'}</p></div>""", unsafe_allow_html=True)
            urls, flags = analyze_links(text)
            words = re.findall(r"\b\w+\b", text)
            caps = sum(1 for w in words if len(w) > 2 and w.isupper())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Links", len(urls)); m2.metric("CAPS words", caps)
            m3.metric("Exclamations", text.count("!")); m4.metric("Words", len(words))
            if st.button("➕ Log to scan history"):
                st.session_state.history.append({"email": text[:80] + ("…" if len(text) > 80 else ""),
                                                 "spam_probability": round(p, 3), "verdict": "SPAM" if is_spam else "HAM"})
        with c2:
            st.plotly_chart(gauge(p), width="stretch")
        ev = evidence(model, text)
        if ev:
            st.subheader("Why this verdict?")
            st.markdown("".join(f'<span class="chip {"s" if v > 0 else "h"}">{w} {v:+.2f}</span>' for w, v in ev),
                        unsafe_allow_html=True)
            st.plotly_chart(hbar([w for w, _ in ev], [v for _, v in ev],
                                 ["#ef4444" if v > 0 else "#22c55e" for _, v in ev],
                                 "Word evidence (red → spam, green → legit)"), width="stretch")
        if urls:
            st.subheader("🔗 Link safety check")
            if flags:
                for u, why in flags:
                    st.markdown(f'<div class="flag">⚠️ <b>{u[:60]}</b> — {why}</div>', unsafe_allow_html=True)
            else:
                st.success("No suspicious patterns found in the links.")
    else:
        st.info("👆 Pick a sample or paste your own email to see the live analysis.")
    if st.session_state.history:
        st.subheader("🕘 Scan history")
        h = pd.DataFrame(st.session_state.history)
        st.dataframe(h, width="stretch", hide_index=True)
        st.download_button("⬇️ Download history (CSV)", h.to_csv(index=False), "scan_history.csv", "text/csv")

# ------------------------------- tab 2 -------------------------------
with tab2:
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{M['acc']:.1%}"); b.metric("Precision", f"{M['pre']:.1%}")
    c.metric("Recall", f"{M['rec']:.1%}"); d.metric("F1 score", f"{M['f1']:.1%}")
    st.caption(f"Evaluated on {M['n_test']} unseen emails (trained on {M['n_train']}).")
    l, r = st.columns(2)
    cm = M["cm"]
    fig = go.Figure(go.Heatmap(z=cm[::-1], x=["Pred. Ham", "Pred. Spam"], y=["Actual Spam", "Actual Ham"],
                               text=cm[::-1], texttemplate="%{text}", colorscale="Purples", showscale=False))
    fig.update_layout(title="Confusion matrix", height=330, margin=dict(t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
    l.plotly_chart(fig, width="stretch")
    roc = go.Figure(go.Scatter(x=M["fpr"], y=M["tpr"], fill="tozeroy", line_color="#6366f1", name="ROC"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Random"))
    roc.update_layout(title=f"ROC curve (AUC = {M['auc']:.3f})", height=330, margin=dict(t=45, b=10),
                      xaxis_title="False positive rate", yaxis_title="True positive rate", paper_bgcolor="rgba(0,0,0,0)")
    r.plotly_chart(roc, width="stretch")

    tf, nb = model.named_steps["tfidf"], model.named_steps["nb"]
    words = np.array(tf.get_feature_names_out())
    lr = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    order = np.argsort(lr)
    l, r = st.columns(2)
    sp_i, ham_i = order[::-1][:10], order[:10]
    l.plotly_chart(hbar(list(words[sp_i]), list(lr[sp_i]), "#ef4444", "Top spam indicators"), width="stretch")
    r.plotly_chart(hbar(list(words[ham_i]), list(-lr[ham_i]), "#22c55e", "Top legit indicators"), width="stretch")
    try:
        from wordcloud import WordCloud
        st.subheader("☁️ Word clouds")
        l, r = st.columns(2)
        for col, y, cmap, name in ((l, 1, "Reds", "Spam"), (r, 0, "Greens", "Legit")):
            corpus = " ".join(clean_text(t) for t in df[df.y == y].text)
            wc = WordCloud(width=700, height=350, background_color=None, mode="RGBA", colormap=cmap,
                           stopwords={"urltoken", "numtoken"}).generate(corpus)
            col.image(wc.to_array(), caption=f"{name} emails", width="stretch")
    except ImportError:
        st.caption("Install `wordcloud` to see word clouds.")

# ------------------------------- tab 3 -------------------------------
with tab3:
    st.write("Upload a CSV of emails and scan them all at once.")
    f = st.file_uploader("CSV file", type="csv", key="batch")
    if f:
        try:
            bd = read_csv_any(f)
            col = st.selectbox("Which column holds the email text?", list(bd.columns),
                               index=next((i for i, c in enumerate(bd.columns) if c.lower() in ("text", "message", "body", "email", "v2")), 0))
            probs = model.predict_proba(bd[col].astype(str))[:, 1]
            bd["spam_probability"] = probs.round(3)
            bd["verdict"] = np.where(probs >= thr, "SPAM", "HAM")
            n_sp = int((probs >= thr).sum())
            x, y = st.columns([1, 2])
            pie = go.Figure(go.Pie(labels=["Spam", "Legit"], values=[n_sp, len(bd) - n_sp], hole=.55,
                                   marker_colors=["#ef4444", "#22c55e"]))
            pie.update_layout(height=300, margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
            x.plotly_chart(pie, width="stretch")
            y.metric("Emails scanned", len(bd)); y.metric("Flagged as spam", n_sp)
            st.dataframe(bd, width="stretch", hide_index=True,
                         column_config={"spam_probability": st.column_config.ProgressColumn("spam_probability", min_value=0, max_value=1)})
            st.download_button("⬇️ Download results", bd.to_csv(index=False), "scan_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not scan this file: {e}")

# ------------------------------- tab 4 -------------------------------
with tab4:
    st.subheader("Bayes' theorem")
    st.latex(r"P(\text{spam}\mid \text{words}) = \frac{P(\text{words}\mid \text{spam})\,P(\text{spam})}{P(\text{words})}")
    st.markdown("""
**Naive** because it assumes words are independent, so the likelihood becomes a product of per-word probabilities:""")
    st.latex(r"P(\text{words}\mid \text{spam}) \approx \prod_i P(w_i \mid \text{spam})")
    st.markdown("""
**Pipeline:** clean text (lowercase, links → `urltoken`, numbers → `numtoken`) → TF-IDF features → Multinomial Naive Bayes → probability.

**Laplace smoothing (α)** adds a small count to every word so an unseen word never makes the probability zero.

**Threshold:** emails with spam probability above it are flagged. A higher threshold means fewer legit emails wrongly flagged.""")
