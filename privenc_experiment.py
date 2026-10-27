"""
Privacy Engineering Encoding Benchmark (synthetic data)
Benchmarks 10 encoding configurations + proposed CALE against 4 adversaries
and measures structured and text utility. Fully synthetic, reproducible.
"""
import hashlib, hmac, json, os, re, secrets, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
from ff3 import FF3Cipher

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
import yaml
CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")))
SEEDS = CFG["seeds"]
N_POP = CFG["population"]["n_people"]
K = CFG["encodings"]["k_anonymity"]
EPS_LO, EPS_HI = CFG["encodings"]["ldp_epsilons"]

# ---------------------------------------------------------------- vocab
FIRST = ["James","Mary","Robert","Patricia","John","Jennifer","Michael","Linda","David","Elizabeth",
 "William","Barbara","Richard","Susan","Joseph","Jessica","Thomas","Sarah","Charles","Karen","Daniel",
 "Lisa","Matthew","Nancy","Anthony","Betty","Mark","Sandra","Donald","Ashley","Steven","Kimberly",
 "Andrew","Emily","Paul","Donna","Joshua","Michelle","Kenneth","Carol","Kevin","Amanda","Brian",
 "Melissa","George","Deborah","Timothy","Stephanie","Ronald","Rebecca","Jason","Laura","Edward",
 "Helen","Jeffrey","Sharon","Ryan","Cynthia","Jacob","Kathleen","Gary","Amy","Nicholas","Angela",
 "Eric","Shirley","Jonathan","Brenda","Stephen","Emma","Larry","Anna","Justin","Pamela","Scott",
 "Nicole","Brandon","Samantha","Benjamin","Katherine","Samuel","Christine","Gregory","Debra",
 "Alexander","Rachel","Patrick","Carolyn","Frank","Janet","Raymond","Maria","Jack","Olivia",
 "Dennis","Heather","Jerry","Diane","Tyler","Julie","Aarav","Priya","Wei","Mei","Carlos","Sofia",
 "Omar","Fatima","Hiroshi","Yuki","Ivan","Olga","Kwame","Amara","Diego","Lucia","Arjun","Ananya"]
LAST = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
 "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
 "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
 "Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores","Green","Adams","Nelson",
 "Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts","Gomez","Phillips","Evans","Turner",
 "Diaz","Parker","Cruz","Edwards","Collins","Reyes","Stewart","Morris","Morales","Murphy","Cook",
 "Rogers","Gutierrez","Ortiz","Morgan","Cooper","Peterson","Bailey","Reed","Kelly","Howard","Ramos",
 "Kim","Cox","Ward","Richardson","Watson","Brooks","Chavez","Wood","James","Bennett","Gray","Mendoza",
 "Ruiz","Hughes","Price","Alvarez","Castillo","Sanders","Patel","Myers","Long","Ross","Foster","Gupta",
 "Sharma","Chen","Wang","Tanaka","Sato","Ivanov","Mensah","Okafor","Silva","Rossi","Muller","Kowalski"]
ZIP3 = ["334","335","336","337","338","339","341","342","344","346","347","349"]
ZIPS = [z3 + f"{i:02d}" for z3 in ZIP3 for i in (10, 18, 26, 33, 41)]  # 60 zips
LANDMARK_A = ["Riverside","Harbor","Cypress","Pinecrest","Lakeview","Oakmont","Seagrape","Palm Grove",
 "Banyan","Coral Bay","Heron Point","Mangrove"]
LANDMARK_B = ["Park","Marina","Library","Mall","Stadium"]
LANDMARK = {z: f"{LANDMARK_A[i // 5]} {LANDMARK_B[i % 5]}" for i, z in enumerate(ZIPS)}
OCC = ["teacher","nurse","software engineer","accountant","electrician","plumber","truck driver",
 "cashier","chef","lawyer","pharmacist","mechanic","farmer","police officer","sales associate",
 "architect","janitor","banker","firefighter","graphic designer"]
OCC_GROUP = {"teacher":"education","nurse":"healthcare","software engineer":"professional",
 "accountant":"professional","electrician":"trades","plumber":"trades","truck driver":"trades",
 "cashier":"service","chef":"service","lawyer":"professional","pharmacist":"healthcare",
 "mechanic":"trades","farmer":"trades","police officer":"service","sales associate":"service",
 "architect":"professional","janitor":"service","banker":"professional","firefighter":"service",
 "graphic designer":"professional"}
OCC_INC = {o: m for o, m in zip(OCC, [58,72,120,78,62,60,55,30,40,140,125,52,45,68,34,95,30,105,64,60])}
DX = {  # name: (icd, meds, category, severity, age_weights[18-34,35-49,50-64,65+])
 "type 2 diabetes":   ("E11.9", ["metformin","glipizide"], "endocrine condition", 0.8, [1,3,5,6]),
 "hypertension":      ("I10",   ["lisinopril","amlodipine"], "circulatory condition", 0.5, [1,3,6,8]),
 "asthma":            ("J45.909",["albuterol","fluticasone"], "respiratory condition", 0.3, [6,4,3,2]),
 "depression":        ("F32.9", ["sertraline","fluoxetine"], "mental health condition", 0.4, [6,5,3,2]),
 "osteoarthritis":    ("M19.90",["meloxicam","naproxen"], "musculoskeletal condition", 0.4, [0.5,2,5,7]),
 "hyperlipidemia":    ("E78.5", ["atorvastatin","rosuvastatin"], "endocrine condition", 0.3, [1,3,5,5]),
 "COPD":              ("J44.9", ["tiotropium","salmeterol"], "respiratory condition", 0.9, [0.2,1,4,6]),
 "chronic kidney disease": ("N18.3",["sevelamer","calcitriol"], "genitourinary condition", 1.0, [0.3,1,3,5]),
}
DXN = list(DX)
FILLER = ["Vitals stable.","Advised hydration.","No acute distress.","Labs ordered.",
          "Follow-up in two weeks.","Discussed diet and exercise.","Patient tolerating treatment.",
          "Referred to specialist.","Reports mild fatigue.",""]
VISITS = ["routine follow-up","new patient","urgent care"]


def ageband(a):
    return np.select([a < 35, a < 50, a < 65], [0, 1, 2], 3)


def decade(a):
    lo = (np.asarray(a) // 10) * 10
    return np.char.add(np.char.add(lo.astype(str), "-"), (lo + 9).astype(str))


# ---------------------------------------------------------------- population
def make_population(rng, n):
    first = rng.choice(FIRST, n); last = rng.choice(LAST, n)
    name = np.char.add(np.char.add(first, " "), last)
    age = np.clip(rng.gamma(9, 5.2, n).astype(int) + 18, 18, 90)
    gender = rng.choice(["F", "M"], n)
    zp = rng.dirichlet(np.ones(60) * 0.8)
    zp = zp / zp.sum()
    zipc = rng.choice(ZIPS, n, p=zp)
    occ = rng.choice(OCC, n)
    ab = ageband(age)
    W = np.array([DX[d][4] for d in DXN], float)  # 8x4
    dx = np.empty(n, dtype=object)
    for b in range(4):
        idx = np.where(ab == b)[0]
        p = W[:, b] / W[:, b].sum()
        dx[idx] = rng.choice(DXN, len(idx), p=p)
    inc = np.array([OCC_INC[o] for o in occ]) * 1000 * np.exp(rng.normal(0, 0.35, n))
    inc = inc * np.where(age > 66, 0.6, 1.0)
    inc = np.clip(inc, 10000, 400000).round(-2)
    sev = np.array([DX[d][3] for d in dx])
    risky = np.isin(occ, ["truck driver","farmer","firefighter","janitor","mechanic"]).astype(float)
    logit = -2.2 + 0.045 * (age - 50) + 1.6 * sev + 0.7 * (inc < 35000) + 0.5 * risky + 0.4 * (gender == "M")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    visit = np.where(y == 1, rng.choice(VISITS, n, p=[0.35, 0.2, 0.45]), rng.choice(VISITS, n, p=[0.6, 0.3, 0.1]))
    acct = np.array([f"{x:010d}" for x in rng.integers(10**9, 10**10, n)])
    med = np.array([DX[d][1][rng.integers(0, 2)] for d in dx])
    has_med = rng.random(n) < CFG["population"]["medication_cue_rate"]
    has_land = rng.random(n) < CFG["population"]["landmark_cue_rate"]
    tpl = rng.integers(0, 3, n)
    fill = rng.choice(FILLER, n)
    df = pd.DataFrame(dict(pid=np.arange(n), name=name, first=first, acct=acct, age=age, gender=gender,
                           zip=zipc, occ=occ, income=inc, dx=dx, y=y, visit=visit, med=med,
                           has_med=has_med, has_land=has_land, tpl=tpl, fill=fill))
    df["zip3"] = df.zip.str[:3]
    df["icd"] = df.dx.map(lambda d: DX[d][0])
    df["dxcat"] = df.dx.map(lambda d: DX[d][2])
    df["occgrp"] = df.occ.map(OCC_GROUP)
    return df


# ---------------------------------------------------------------- text rendering
def render(r, E):
    """E: dict of surface forms for entities in this record."""
    pron = "She" if r.gender == "F" else "He"
    gw = "female" if r.gender == "F" else "male"
    occp = (f"retired {E['occ']}" if r.age >= 67 else E["occ"])
    med_s = f"{pron} is currently taking {r.med}. " if r.has_med else ""
    land_s = f"{pron} lives near {LANDMARK[r.zip]}. " if r.has_land else ""
    if r.tpl == 0:
        t = (f"{E['name']}, a {E['age']}-year-old {occp} from ZIP {E['zip']}, seen for a {r.visit} visit. "
             f"Diagnosis: {E['dx']} ({E['icd']}). {med_s}{land_s}Account {E['acct']}. {r.fill}")
    elif r.tpl == 1:
        t = (f"Patient {E['name']} (age {E['age']}, {gw}) presented for a {r.visit} visit regarding {E['dx']}. "
             f"Works as a {occp}. {land_s}{med_s}Billing account {E['acct']}. {r.fill}")
    else:
        t = (f"{r.visit.capitalize()} visit. {E['name']}, {E['age']} y/o {gw} {occp}, resides in {E['zip']}. "
             f"Assessed for {E['dx']}, code {E['icd']}. {med_s}{land_s}{r.fill}")
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- encoders
class Keys:
    def __init__(self, seed):
        r = np.random.default_rng(seed + 999)
        self.hmac = bytes(r.integers(0, 256, 32, dtype=np.uint8))
        self.fpe_key = bytes(r.integers(0, 256, 16, dtype=np.uint8)).hex()
        self.fpe_tweak = bytes(r.integers(0, 256, 7, dtype=np.uint8)).hex()
        self.vault = {}
        self.vr = np.random.default_rng(seed + 4242)
        self.fpe_num = FF3Cipher(self.fpe_key, self.fpe_tweak, radix=10)
        self.fpe_alpha = FF3Cipher.withCustomAlphabet(self.fpe_key, self.fpe_tweak, "abcdefghijklmnopqrstuvwxyz")
        self.fpe_an = FF3Cipher.withCustomAlphabet(self.fpe_key, self.fpe_tweak, "0123456789abcdefghijklmnopqrstuvwxyz")
        self.cache = {}

    def token(self, kind, v):  # vault tokenization: random, consistent, reversible via vault
        k = (kind, v)
        if k not in self.vault:
            self.vault[k] = f"{kind}{self.vr.integers(0, 16**6):06x}"
        return self.vault[k]

    def hm(self, kind, v, salt=b""):
        return kind + hmac.new(self.hmac + salt, f"{kind}|{v}".encode(), hashlib.sha256).hexdigest()[:8]

    def fpe(self, kind, v):
        k = ("fpe", kind, v)
        if k in self.cache:
            return self.cache[k]
        if kind == "num":
            o = self.fpe_num.encrypt(v)
        elif kind == "name":
            s = re.sub(r"[^a-z]", "", v.lower()).ljust(6, "x")
            o = self.fpe_alpha.encrypt(s).capitalize()
        else:  # icd, alnum padded to 6 to satisfy the 1e6 minimum-domain rule of FF3-1
            s = re.sub(r"[^0-9a-z]", "", v.lower()).ljust(6, "0")
            o = self.fpe_an.encrypt(s).upper()
        self.cache[k] = o
        return o


def rr(rng, vals, cats, eps):
    """k-ary randomized response (local DP)."""
    vals = np.asarray(vals, dtype=object); k = len(cats)
    keep = rng.random(len(vals)) < np.exp(eps) / (np.exp(eps) + k - 1)
    out = vals.copy()
    flip = np.where(~keep)[0]
    cats = np.asarray(cats, dtype=object)
    for i in flip:
        others = cats[cats != vals[i]]
        out[i] = others[rng.integers(0, len(others))]
    return out


def k_suppress(df, cols, k=5):
    sz = df.groupby(cols)[cols[0]].transform("size")
    return df[sz >= k].copy()


def fit_synth(rng, d, n):
    """Degree-1 Bayesian-network style synthesizer fit on the member table."""
    out = pd.DataFrame(index=range(n))
    out["age"] = np.clip(rng.choice(d.age.values, n) + rng.integers(-2, 3, n), 18, 90)
    out["gender"] = rng.choice(d.gender.values, n)
    out["zip"] = rng.choice(d.zip.values, n)
    ab = ageband(out.age.values); dab = ageband(d.age.values)
    occ = np.empty(n, dtype=object); dx = np.empty(n, dtype=object)
    for b in range(4):
        idx = np.where(ab == b)[0]
        pool = d[dab == b]
        occ[idx] = rng.choice(pool.occ.values, len(idx))
        dx[idx] = rng.choice(pool.dx.values, len(idx))
    out["occ"] = occ; out["dx"] = dx
    li = np.log(d.income.values)
    stats = pd.DataFrame({"o": d.occ.values, "l": li}).groupby("o").l.agg(["mean", "std"])
    out["income"] = np.exp(stats.loc[out.occ, "mean"].values + rng.normal(0, 1, n) * stats.loc[out.occ, "std"].values).round(-2)
    # label | (dx, ageband, gender) empirical rates with Laplace smoothing
    g = pd.DataFrame({"dx": d.dx.values, "ab": dab, "g": d.gender.values, "y": d.y.values})
    rate = g.groupby(["dx", "ab", "g"]).y.agg(["sum", "count"])
    key = pd.MultiIndex.from_arrays([out.dx.values, ab, out.gender.values])
    s = rate.reindex(key)
    p = ((s["sum"].fillna(0) + 1) / (s["count"].fillna(0) + 2)).values
    out["y"] = (rng.random(n) < p).astype(int)
    out["zip3"] = out.zip.str[:3]; out["occgrp"] = out.occ.map(OCC_GROUP)
    return out


# ---------------------------------------------------------------- feature views
NUM = ["age", "income"]


def feat(df, mode):
    d = pd.DataFrame(index=df.index)
    if mode in ("gen",):
        d["age"] = (df.age // 10) * 10 + 5
        d["income"] = (df.income // 25000) * 25000 + 12500
        d["gender"] = df.gender; d["loc"] = df.zip3; d["occ"] = df.occgrp; d["dx"] = df.dx.map(lambda x: DX[x][2])
    elif mode == "mask":
        d["age"] = df.age; d["income"] = df.income; d["gender"] = df.gender
        d["loc"] = df.zip3; d["occ"] = df.occ; d["dx"] = df.dx
    else:
        d["age"] = df.age; d["income"] = df.income; d["gender"] = df.gender
        d["loc"] = df.zip; d["occ"] = df.occ; d["dx"] = df.dx
    return d


def train_eval_rf(Xtr_df, ytr, Xte_df, yte, Xmem_df, ymem, seed):
    cats = ["gender", "loc", "occ", "dx"]
    enc = OneHotEncoder(handle_unknown="ignore").fit(Xtr_df[cats].astype(str))
    def M(X):
        return hstack([enc.transform(X[cats].astype(str)), csr_matrix(X[NUM].values.astype(float) / [100, 1e5])]).tocsr()
    rf = RandomForestClassifier(n_estimators=CFG["attacks"]["rf_trees"], min_samples_leaf=1, max_features="sqrt", n_jobs=-1, random_state=seed)
    rf.fit(M(Xtr_df), ytr)
    p_te = rf.predict_proba(M(Xte_df))[:, 1]
    auc = roc_auc_score(yte, p_te)
    # membership inference: confidence on the true label, members vs non-members
    p_mem = rf.predict_proba(M(Xmem_df))[:, 1]
    c_mem = np.where(ymem == 1, p_mem, 1 - p_mem)
    c_non = np.where(yte == 1, p_te, 1 - p_te)
    mia = roc_auc_score(np.r_[np.ones(len(c_mem)), np.zeros(len(c_non))], np.r_[c_mem, c_non])
    return auc, mia


# ---------------------------------------------------------------- attacks
def linkage_attack(release, aux, keys_rel, keys_aux, members_pid, base_acc, dx_col="dx"):
    """Unique-match re-identification + attribute (dx) disclosure via aux file."""
    a = aux[aux.pid.isin(members_pid)].copy()
    r = release.copy(); r["_rid"] = np.arange(len(r))
    if keys_rel is None:
        return 0.0, 0.0
    m = a[["pid"] + keys_aux].merge(r[keys_rel + ["_rid", "_truepid", dx_col]], left_on=keys_aux, right_on=keys_rel, how="left")
    cnt = m.groupby("pid")._rid.transform("count")
    uniq = m[(cnt == 1)]
    reid = (uniq._truepid == uniq.pid).sum() / len(a)
    # attribute disclosure: majority dx among matches, else population-level fallback
    maj = m.dropna(subset=["_rid"]).groupby("pid")[dx_col].agg(lambda s: s.value_counts().index[0])
    pred = a.pid.map(maj)
    fallback = a.age.map(lambda x: base_acc["map"][int(ageband(np.array([x]))[0])])
    pred = pred.fillna(fallback)
    truth = a.dx if dx_col == "dx" else a.dx.map(lambda x: DX[x][2])
    acc = (pred.values == truth.values).mean()
    adv = acc - (base_acc["acc"] if dx_col == "dx" else base_acc["acc_cat"])
    return float(reid), float(max(adv, 0.0))


def text_attack(texts, labels, rng_seed, attack_frac=None):
    attack_frac = attack_frac or CFG["attacks"]["text_attack_train_fraction"]
    """Two learned contextual-inference adversaries (word and char n-gram); returns max advantage."""
    idx = np.arange(len(texts))
    strat = labels if pd.Series(labels).value_counts().min() >= 5 else None
    tr, te = train_test_split(idx, train_size=attack_frac, random_state=rng_seed, stratify=strat)
    base = pd.Series(labels[tr]).value_counts(normalize=True).iloc[0]
    best = 0.0
    for vec in (TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)):
        Xtr = vec.fit_transform(texts[tr]); Xte = vec.transform(texts[te])
        clf = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, labels[tr])
        acc = accuracy_score(labels[te], clf.predict(Xte))
        best = max(best, (acc - base) / (1 - base))  # normalized advantage in [0,1]
    return float(max(best, 0.0))


def text_utility(texts, labels, seed):
    """Downstream NLP task: predict high-cost outcome from the note (AUC)."""
    tr, te = train_test_split(np.arange(len(texts)), test_size=0.5, random_state=seed, stratify=labels)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(texts[tr]), labels[tr])
    return float(roc_auc_score(labels[te], clf.predict_proba(vec.transform(texts[te]))[:, 1]))


def leakage_guided_scrub(texts, protected, seed, delta=CFG["cale"]["delta"], step=CFG["cale"]["m"], max_iter=CFG["cale"]["max_rounds"], cal_texts=None, cal_protected=None):
    """CALE text stage (Adversary agent A_int): adversary-in-the-loop scrubbing of high-leakage cue tokens.
    v2: A_int is fitted and validated ONLY on the controller's calibration notes (cal_texts), which are
    disjoint from the released notes that the external adversaries attack. The learned scrub list is then
    applied to the release. Stops when A_int's normalized advantage on every protected attribute < delta."""
    rel_texts = texts
    if cal_texts is not None:
        texts, protected = cal_texts, cal_protected
    idx = np.arange(len(texts))
    fit_i, val_i = train_test_split(idx, test_size=0.5, random_state=seed + 7)
    scrub = set(); hist = []; round_t = []
    tok_re = re.compile(r"[A-Za-z][A-Za-z\-]+")
    def apply(ts):
        if not scrub:
            return ts
        pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, scrub), key=len, reverse=True)) + r")\b", re.I)
        return np.array([re.sub(r"\s+", " ", pat.sub("[CUE]", t)) for t in ts], dtype=object)
    for it in range(max_iter):
        tr0 = time.perf_counter()
        cur = apply(texts)
        advs = []; tops = []
        for lab in protected:
            vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2, token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]+\b")
            Xf = vec.fit_transform(cur[fit_i]); Xv = vec.transform(cur[val_i])
            clf = LogisticRegression(max_iter=2000, C=4.0).fit(Xf, lab[fit_i])
            base = pd.Series(lab[fit_i]).value_counts(normalize=True).iloc[0]
            acc = accuracy_score(lab[val_i], clf.predict(Xv))
            advs.append(max((acc - base) / (1 - base), 0))
            w = np.abs(clf.coef_).max(axis=0)
            vocab = np.array(vec.get_feature_names_out())
            order = np.argsort(-w)
            tops.append([v for v in vocab[order] if v.lower() not in {"cue", "redacted"} and v.lower() not in scrub][:step])
        hist.append(max(advs))
        round_t.append(time.perf_counter() - tr0)
        if max(advs) < delta:
            break
        for tl, a in zip(tops, advs):
            if a >= delta:
                scrub.update(t.lower() for t in tl)
    ta = time.perf_counter(); out = apply(rel_texts); t_apply = time.perf_counter() - ta
    return out, sorted(scrub), dict(hist=hist, round_s=round_t, apply_s=t_apply, n_cal=len(texts))


# ---------------------------------------------------------------- main experiment
METHODS = ["Raw (no encoding)", "Redaction", "Partial masking", "Tokenization (vault)", "Keyed hashing (HMAC)",
           "FPE (FF3-1)", "Generalization (k=5)", "Local DP (eps=1)", "Local DP (eps=4)", "Synthetic data (BN)",
           "CALE (proposed)", "CALE-S (proposed)"]


def run(seed):
    rng = np.random.default_rng(seed)
    pop = make_population(rng, N_POP)
    mem = pop.iloc[:10000].copy(); non = pop.iloc[10000:20000].copy()
    r2_ids = np.r_[rng.choice(10000, 5000, replace=False), rng.choice(np.arange(20000, 30000), 5000, replace=False)]
    rel2 = pop.iloc[r2_ids].copy()
    aux = pop.sample(frac=CFG["population"]["auxiliary_fraction"], random_state=seed).copy()  # public auxiliary file (voter-roll style)
    keys = Keys(seed)
    # attacker's population-level knowledge: best dx guess per age band, learned on non-members
    ab_non = ageband(non.age.values)
    mp = {b: non.dx[ab_non == b].value_counts().index[0] for b in range(4)}
    mpc = {b: non.dxcat[ab_non == b].value_counts().index[0] for b in range(4)}
    abm = ageband(mem.age.values)
    base = {"map": mp, "acc": float((mem.dx.values == np.vectorize(mp.get)(abm)).mean()),
            "acc_cat": float((mem.dxcat.values == np.vectorize(mpc.get)(abm)).mean())}
    res = {}; extra = {}
    entity_redact = lambda r: {"name": "[PERSON]", "age": "[AGE]", "zip": "[LOCATION]", "dx": "[CONDITION]",
                               "icd": "[CODE]", "acct": "[ACCOUNT]", "occ": r.occ}
    for M in METHODS:
        t0 = time.time(); tp0 = time.perf_counter()
        relA = mem.copy(); relA["_truepid"] = relA.pid
        relB = rel2.copy(); relB["_truepid"] = relB.pid
        fmode = "raw"; trainX = None; trainy = mem.y.values; keys_rel = keys_aux = None; dxcol = "dx"
        link_cols = None
        reversible = False
        if M == "Raw (no encoding)":
            ent = lambda r: {"name": r.name, "age": r.age, "zip": r.zip, "dx": r.dx, "icd": r.icd, "acct": r.acct, "occ": r.occ}
            keys_rel = keys_aux = ["name", "age", "gender", "zip"]
            link_cols = ["name", "age", "gender", "zip", "occ", "dx"]
        elif M == "Redaction":
            ent = entity_redact
            keys_rel = keys_aux = ["age", "gender", "zip", "occ"]
            link_cols = ["age", "gender", "zip", "occ", "dx"]
        elif M == "Partial masking":
            ent = lambda r: {"name": r.first[0] + "*** " + r.name.split()[1][0] + "***", "age": r.age, "zip": r.zip3 + "**",
                             "dx": "[MASKED]", "icd": r.icd[0] + "**.*", "acct": "******" + r.acct[-4:], "occ": r.occ}
            for R in (relA, relB):
                R["ini"] = R["first"].str[0] + R["name"].str.split().str[1].str[0]
            aux = aux.assign(ini=aux["first"].str[0] + aux["name"].str.split().str[1].str[0])
            keys_rel = keys_aux = ["ini", "age", "gender", "zip3", "occ"]
            link_cols = ["ini", "age", "gender", "zip3", "occ", "dx", "acct4"]
            for R in (relA, relB):
                R["acct4"] = R.acct.str[-4:]
            fmode = "mask"
        elif M == "Tokenization (vault)":
            reversible = True
            ent = lambda r: {"name": keys.token("P", r.name + r.acct), "age": r.age, "zip": keys.token("Z", r.zip),
                             "dx": keys.token("D", r.dx), "icd": keys.token("C", r.icd), "acct": keys.token("A", r.acct), "occ": r.occ}
            for R in (relA, relB):
                R["tok"] = [keys.token("P", n + a) for n, a in zip(R["name"], R.acct)]
            keys_rel = keys_aux = ["age", "gender", "zip", "occ"]
            link_cols = ["tok"]
        elif M == "Keyed hashing (HMAC)":
            ent = lambda r: {"name": keys.hm("P", r.name + r.acct), "age": r.age, "zip": keys.hm("Z", r.zip),
                             "dx": keys.hm("D", r.dx), "icd": keys.hm("C", r.icd), "acct": keys.hm("A", r.acct), "occ": r.occ}
            for R in (relA, relB):
                R["tok"] = [keys.hm("P", n + a) for n, a in zip(R["name"], R.acct)]
            keys_rel = keys_aux = ["age", "gender", "zip", "occ"]
            link_cols = ["tok"]
        elif M == "FPE (FF3-1)":
            reversible = True
            # 5-digit ZIP (domain 1e5) is below the FF3-1 minimum domain of 1e6, so it falls back to redaction
            ent = lambda r: {"name": keys.fpe("name", r.first) + " " + keys.fpe("name", r.name.split()[1]), "age": r.age,
                             "zip": "[LOCATION]", "dx": "[CONDITION]", "icd": keys.fpe("icd", r.icd),
                             "acct": keys.fpe("num", r.acct), "occ": r.occ}
            for R in (relA, relB):
                R["tok"] = [keys.fpe("num", a) for a in R.acct]
            keys_rel = keys_aux = ["age", "gender", "zip", "occ"]
            link_cols = ["tok"]
        elif M == "Generalization (k=5)":
            ent = lambda r: {"name": "[PERSON]", "age": str(decade(np.array([r.age]))[0]), "zip": r.zip3 + "xx",
                             "dx": r.dxcat, "icd": r.icd[0] + "-chapter", "acct": "[ACCOUNT]", "occ": r.occgrp + " worker"}
            for R in (relA, relB):
                R["dec"] = (R.age // 10); R["dxcat_"] = R.dxcat
            aux = aux.assign(dec=aux.age // 10)
            relA = k_suppress(relA, ["dec", "gender", "zip3", "occgrp"], K)
            relB = k_suppress(relB, ["dec", "gender", "zip3", "occgrp"], K)
            keys_rel = keys_aux = ["dec", "gender", "zip3", "occgrp"]
            link_cols = ["dec", "gender", "zip3", "occgrp", "dxcat_"]
            fmode = "gen"; dxcol = "dxcat"
        elif M.startswith("Local DP"):
            eps = EPS_LO if "eps=1" in M else EPS_HI
            for R in (relA, relB):
                R["gender"] = rr(rng, R.gender.values, ["F", "M"], eps)
                R["zip"] = rr(rng, R.zip.values, ZIPS, eps)
                R["occ"] = rr(rng, R.occ.values, OCC, eps)
                R["dx"] = rr(rng, R.dx.values, DXN, eps)
                R["age"] = np.clip(np.round(R.age + rng.laplace(0, 72 / eps, len(R))), 18, 90).astype(int)
                R["income"] = np.clip(R.income + rng.laplace(0, 390000 / eps, len(R)), 10000, 400000)
                R["y"] = rr(rng, R.y.values, [0, 1], eps).astype(int)
                R["zip3"] = R.zip.str[:3]
            trainy = relA.y.values
            ent = entity_redact  # no DP mechanism for free text in mainstream tools: entity-redaction fallback
            keys_rel = keys_aux = ["age", "gender", "zip", "occ"]
            link_cols = ["age", "gender", "zip", "occ", "dx"]
        elif M == "Synthetic data (BN)":
            syn = fit_synth(rng, mem, len(mem)); syn["_truepid"] = -1
            relA = syn; trainy = syn.y.values
            synB = fit_synth(rng, rel2, len(rel2)); synB["_truepid"] = -1; relB = synB
            ent = entity_redact
            keys_rel = keys_aux = ["age", "gender", "zip", "occ"]
            link_cols = ["age", "gender", "zip", "occ", "dx"]
        elif M == "CALE-S (proposed)":
            syn = fit_synth(rng, mem, len(mem)); syn["_truepid"] = -1
            relA = syn; trainy = syn.y.values
            synB = fit_synth(rng, rel2, len(rel2)); synB["_truepid"] = -1; relB = synB
            ent = entity_redact
            keys_rel = keys_aux = ["age", "gender", "zip", "occ"]
            link_cols = ["age", "gender", "zip", "occ", "dx"]
        elif M == "CALE (proposed)":
            salt_a, salt_b = b"release-A", b"release-B"   # per-release salt: unlinkable pseudonyms
            for R, s in ((relA, salt_a), (relB, salt_b)):
                R["tok"] = [keys.hm("P", n + a, s) for n, a in zip(R["name"], R.acct)]
                R["dec"] = R.age // 10; R["dxcat_"] = R.dxcat
            aux = aux.assign(dec=aux.age // 10)
            relA = k_suppress(relA, ["dec", "gender", "zip3", "occgrp"], K)
            relB = k_suppress(relB, ["dec", "gender", "zip3", "occgrp"], K)
            ent = entity_redact
            keys_rel = keys_aux = ["dec", "gender", "zip3", "occgrp"]
            link_cols = ["tok", "dec", "gender", "zip3", "occgrp", "dxcat_"]
            fmode = "gen"; dxcol = "dxcat"
        t_struct = time.perf_counter() - tp0
        # ---------- structured utility + membership inference
        tev0 = time.perf_counter()
        SYN = M in ("Synthetic data (BN)", "CALE-S (proposed)")
        if not (M.startswith("Local DP") or SYN):
            trainy = relA.y.values
        if M.startswith("Local DP"):
            trX = feat(relA, "raw"); teX = feat(non, "raw"); memX = feat(mem, "raw")
        elif SYN:
            trX = feat(relA, "raw"); teX = feat(non, "raw"); memX = feat(mem, "raw")
        else:
            trX = feat(relA, fmode); teX = feat(non, fmode)
            memX = feat(relA, fmode)
        ymem = mem.y.values if SYN or M.startswith("Local DP") else relA.y.values
        auc, mia = train_eval_rf(trX, trainy, teX, non.y.values, memX, ymem, seed)
        # ---------- linkage (re-identification + attribute disclosure)
        A = aux.copy()
        if SYN:
            reid, adis = linkage_attack(relA.assign(_truepid=-1), A, keys_rel, keys_aux, mem.pid.values, base, "dx")
        else:
            reid, adis = linkage_attack(relA, A, keys_rel, keys_aux, mem.pid.values, base, dxcol)
        # ---------- cross-release linkability (no auxiliary data)
        both = set(relA._truepid) & set(relB._truepid) - {-1}
        if len(both) == 0 or SYN:
            linkab = 0.0
        else:
            ra = relA[link_cols + ["_truepid"]].rename(columns={"_truepid": "pa"})
            rb = relB[link_cols + ["_truepid"]].rename(columns={"_truepid": "pb"})
            if "tok" in link_cols and M != "CALE (proposed)":
                mm = ra[["tok", "pa"]].merge(rb[["tok", "pb"]], on="tok")
            else:
                cols = [c for c in link_cols if c != "tok"]
                mm = ra.merge(rb, on=cols)
            ca = mm.groupby("pa").pb.transform("count")
            mm = mm[ca == 1]
            linkab = float((mm.pa == mm.pb).sum() / len(both))
        t_eval = time.perf_counter() - tev0
        # ---------- free text encoding (render with the configuration's entity encoding)
        tt0 = time.perf_counter()
        texts = np.array([render(r, ent(r)) for r in mem.itertuples()], dtype=object)
        t_text = time.perf_counter() - tt0
        lab_dx = mem.dx.values.astype(str); lab_loc = mem.zip.values.astype(str)
        scrub = None; t_scrub = 0.0
        if M.startswith("CALE"):
            # controller calibration sample: 5,000 non-member records, encoded the same way, disjoint from the release
            cal = non.iloc[:CFG["cale"]["calibration_notes"]]
            cal_texts = np.array([render(r, ent(r)) for r in cal.itertuples()], dtype=object)
            ts0 = time.perf_counter()
            texts, scrub, info = leakage_guided_scrub(texts, None, seed, cal_texts=cal_texts,
                                                      cal_protected=[cal.dx.values.astype(str), cal.zip.values.astype(str)])
            t_scrub = time.perf_counter() - ts0
            extra[M + "|scrub"] = scrub; extra[M + "|scrub_info"] = info
        tev1 = time.perf_counter()
        tdx = text_attack(texts, lab_dx, seed)
        tloc = text_attack(texts, lab_loc, seed + 1)
        tutil = text_utility(texts, mem.y.values, seed)
        t_eval += time.perf_counter() - tev1
        res[M] = dict(utility_auc=auc, mia_auc=mia, reid=reid, attr_disc=adis, linkability=linkab,
                      text_dx_adv=tdx, text_loc_adv=tloc, text_util=tutil, reversible=reversible,
                      released=len(relA), example=texts[3], t_struct=t_struct, t_text=t_text, t_scrub=t_scrub,
                      t_eval=t_eval)
        print(f"[seed {seed}] {M:24s} auc={auc:.3f} mia={mia:.3f} reid={reid:.3f} adis={adis:.3f} link={linkab:.3f} "
              f"tdx={tdx:.3f} tloc={tloc:.3f} tutil={tutil:.3f} n={len(relA)} ({time.time()-t0:.1f}s)", flush=True)
    return res, extra


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    allr = {}; extras = {}
    for s in SEEDS:
        allr[s], extras[s] = run(s)
    json.dump({"results": {str(k): v for k, v in allr.items()}, "extra": {str(k): v for k, v in extras.items()}},
              open(f"{OUT}/raw_results.json", "w"), indent=1, default=str)
    print("done")
