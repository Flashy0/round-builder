# app.py
import re
import pandas as pd
import streamlit as st

EXCEL_PATH = "Quiz Prep.xlsx"
ALLOWED_RANKS = {"menu", "certified", "junior", "senior"}

MIX_RULES = {
    "menu":      {"menu": 4},
    "certified": {"certified": 2, "menu": 2},
    "junior":    {"junior": 2, "certified": 1, "menu": 1},
    "senior":    {"senior": 2, "junior": 2, "certified": 1, "menu": 1},  # total 6
}

# ---------- Parsers & helpers (your code, slightly adapted) ----------
def normalize_ingredient(raw_ing: str) -> str:
    no_qty = re.sub(r"\([^)]*\)", "", str(raw_ing))
    cleaned = re.sub(r"\s+", " ", no_qty).strip().lower()
    cleaned = re.sub(r"\bj\b", "juice", cleaned)
    return cleaned

def parse_ingredients_cell_set(cell: str) -> set:
    if not isinstance(cell, str):
        return set()
    ing = set()
    for part in (p for p in cell.split(",") if p.strip()):
        for alt in (s for s in re.split(r"\bOR\b|/|;", part, flags=re.IGNORECASE) if s.strip()):
            name = normalize_ingredient(alt)
            if name:
                ing.add(name)
    return ing

def parse_ingredients_cell_list(cell: str) -> list:
    if not isinstance(cell, str):
        return []
    seen, ordered = set(), []
    for part in (p for p in cell.split(",") if p.strip()):
        for alt in (s for s in re.split(r"\bOR\b|/|;", part, flags=re.IGNORECASE) if s.strip()):
            name = normalize_ingredient(alt)
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered

METHOD_ORDER = {"build": 0, "stir": 1, "shake": 2}

def get_method_priority(method_str):
    if not isinstance(method_str, str):
        return 99
    m = method_str.lower()
    if "shake" in m: return METHOD_ORDER["shake"]
    if "stir"  in m: return METHOD_ORDER["stir"]
    if "build" in m: return METHOD_ORDER["build"]
    return 99

def get_method_complexity(method_str):
    if not isinstance(method_str, str):
        return 999
    chunks = [c.strip() for c in re.split(r"[,&]| and ", method_str.lower()) if c.strip()]
    return len(chunks) if chunks else 999

MISC_KEYWORDS = (
    "mint", "bitters", "ango bitters", "angostura", "orange bitters",
    "foamer", "egg white", "saline", "salt solution", "saline solution",
    "tea bag", "zest", "peel", "aroma", "spray", "mist",
    "puree", "bitter", "coco cream", "cucumber", "slices", "wedges", "premix", "jam ()", "nutella mix", "egg"
)
PASSIONFRUIT_PUREE_PATTERNS = (r"passion ?fruit puree", r"pf puree")

NON_ALC_KEYWORDS = (
    "juice", "cordial", "syrup", "gomme", "sugar", "honey", "agave", "maple",
    "cream", "milk", "coconut milk",
    "coffee", "espresso", "cold brew",
    "soda", "soda water", "sparkling", "sparkling water",
    "tonic", "ginger beer", "ginger ale", "lemonade", "cola", "pepsi", "half & half"
)
NON_ALC_TOPPERS = (
    "soda", "soda water", "sparkling", "sparkling water",
    "tonic", "ginger beer", "ginger ale", "lemonade", "cola"
)

def is_match(name: str, patterns) -> bool:
    n = name.lower()
    return any(p in n for p in patterns)

def is_regex_match(name: str, regex_patterns) -> bool:
    n = name.lower()
    return any(re.search(rp, n) for rp in regex_patterns)

def classify_ingredient(name: str) -> str:
    if is_regex_match(name, PASSIONFRUIT_PUREE_PATTERNS):
        return "non_alc"
    if is_match(name, MISC_KEYWORDS):
        return "misc"
    if is_match(name, NON_ALC_KEYWORDS):
        return "non_alc"
    return "alc"

def split_for_order_of_ops(ingredient_list: list[str], method_str: str):
    misc, non_alc, alc, toppers = [], [], [], []
    m = (method_str or "").lower()
    for ing in ingredient_list:
        cat = classify_ingredient(ing)
        if cat == "misc":
            misc.append(ing)
        elif cat == "non_alc":
            if any(tok in ing for tok in NON_ALC_TOPPERS) and "top" in m:
                toppers.append(ing)   # hold to the very end
            else:
                non_alc.append(ing)
        else:
            alc.append(ing)
    return misc, non_alc, alc, toppers

def ing_overlap(drink_to_ingredients, a_drink, b_drink):
    return len(drink_to_ingredients.get(a_drink, set()) & drink_to_ingredients.get(b_drink, set()))

def cluster_by_ingredients(rows_in_same_method: pd.DataFrame, drink_to_ingredients: dict) -> pd.DataFrame:
    if len(rows_in_same_method) <= 1:
        return rows_in_same_method
    remaining = list(rows_in_same_method.itertuples(index=False))
    def total_overlap(d, others):
        return sum(ing_overlap(drink_to_ingredients, d.Drink, o.Drink) for o in others if o.Drink != d.Drink)
    start = max(remaining, key=lambda d: total_overlap(d, remaining))
    ordered = [start]; remaining.remove(start)
    while remaining:
        last = ordered[-1]
        next_choice = max(remaining, key=lambda d: ing_overlap(drink_to_ingredients, last.Drink, d.Drink))
        ordered.append(next_choice); remaining.remove(next_choice)
    cols = rows_in_same_method.columns
    return pd.DataFrame([[getattr(r, c) for c in cols] for r in ordered], columns=cols)

def normalize_rank(s: str) -> str:
    return str(s).strip().lower()

def sample_by_mix(df: pd.DataFrame, chosen_rank_key: str) -> pd.DataFrame:
    if chosen_rank_key not in MIX_RULES:
        raise ValueError(f"Unknown rank: {chosen_rank_key}")
    df = df.dropna(subset=["Drink"]).copy()
    df["RankKey"] = df["Rank"].apply(normalize_rank)
    required = MIX_RULES[chosen_rank_key]
    picks = []
    for bucket_rank, need in required.items():
        pool = df[df["RankKey"] == bucket_rank]
        have = len(pool)
        if have < need:
            raise ValueError(f"Not enough drinks for bucket '{bucket_rank.title()}': need {need}, have {have}.")
        picks.append(pool.sample(n=need, replace=False))
    return pd.concat(picks, ignore_index=True)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Round Builder", layout="centered")
st.title("🍸 Round Builder (uses local Quiz Prep.xlsx)")

st.sidebar.markdown("**Options**")
rank_choice = st.sidebar.selectbox("Choose rank", ["Menu","Certified","Junior","Senior"])
generate = st.sidebar.button("Generate round")

# Load local workbook and check
@st.cache_data(ttl=3600)
def load_local_sheet(path):
    return pd.read_excel(path)

try:
    df = load_local_sheet(EXCEL_PATH)
except Exception as e:
    st.error(f"Could not load {EXCEL_PATH}: {e}")
    st.stop()

required_cols = {"Drink", "Ingredients", "Method", "Rank", "Glass", "Ice", "Garnish"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns in {EXCEL_PATH}: {', '.join(sorted(missing))}")
    st.stop()

if generate:
    chosen_key = rank_choice.lower()
    try:
        sampled_rows = sample_by_mix(df, chosen_key)
    except Exception as e:
        st.error(str(e))
        st.stop()

    drink_to_ingredients_set = {row.Drink: parse_ingredients_cell_set(row.Ingredients)
                                for row in sampled_rows.itertuples(index=False)}
    drink_to_ingredients_list = {row.Drink: parse_ingredients_cell_list(row.Ingredients)
                                 for row in sampled_rows.itertuples(index=False)}

    st.subheader("Your round")
    for i, row in enumerate(sampled_rows.itertuples(index=False), start=1):
        st.write(f"**{i}. {row.Drink}** — {row.Ingredients}")

    # common ingredients
    ingredient_to_drinks = {}
    for drink, ings in drink_to_ingredients_set.items():
        for ing in ings:
            ingredient_to_drinks.setdefault(ing, []).append(drink)
    common_ingredients = {ing: drinks for ing, drinks in ingredient_to_drinks.items() if len(drinks) >= 2}

    st.subheader("Common ingredients")
    if common_ingredients:
        for ing, drinks in sorted(common_ingredients.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            st.write(f"- **{ing.capitalize()}** appears in: {', '.join(drinks)}")
    else:
        st.write("No common ingredients found between these drinks.")
    st.write("")

    # mat order
    sampled_rows = sampled_rows.assign(
        MethodPriority=sampled_rows["Method"].apply(get_method_priority),
        MethodComplexity=sampled_rows["Method"].apply(get_method_complexity),
    )
    base_sorted = sampled_rows.sort_values(by=["MethodPriority", "MethodComplexity", "Drink"]).reset_index(drop=True)

    final_chunks = []
    for _, group_df in base_sorted.groupby("MethodPriority", sort=True):
        clustered = cluster_by_ingredients(group_df, drink_to_ingredients_set)
        final_chunks.append(clustered)
    final_mat_order_df = pd.concat(final_chunks, ignore_index=True)

    st.subheader("Suggested mat order (left → right)")
    for i, row in enumerate(final_mat_order_df.itertuples(index=False), start=1):
        st.write(f"{i}. **{row.Drink}** — {row.Method}")

    st.subheader("Order of operations per drink")
    details = []
    for row in final_mat_order_df.itertuples(index=False):
        ing_list   = drink_to_ingredients_list.get(row.Drink, [])
        method_str = getattr(row, "Method", "") or ""
        glass_str  = getattr(row, "Glass", "") or "Glassware N/A"
        ice_str    = getattr(row, "Ice", "") or "Ice N/A"
        garnish_str= getattr(row, "Garnish", "") or "Garnish N/A"

        misc, non_alc, alc, toppers = split_for_order_of_ops(ing_list, method_str)

        st.markdown(f"**{row.Drink}** — Glass: _{glass_str}_ — Ice: _{ice_str}_ — Garnish: _{garnish_str}_ — Method: _{method_str}_")
        def pretty(items): return ", ".join(items) if items else "—"
        st.write(f"- Misc: {pretty(misc)}")
        st.write(f"- Non-alcoholic: {pretty(non_alc)}")
        st.write(f"- Alcoholic: {pretty(alc)}")
        if toppers:
            st.write(f"- Top (last): {pretty(toppers)}")

    # download CSV
    st.write("")
    for row in final_mat_order_df.itertuples(index=False):
        # collect details for CSV if required (lightweight)
        pass
    st.success("Round generated. Add this page to your Home Screen for quick access.")

