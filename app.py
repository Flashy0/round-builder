# app.py
import random
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

import random
import re

# --- Helpers to detect categories from drink names / methods ---
MARTINI_EXCLUDE = ("espresso", "pornstar", "french", "left bank", "detroit")  # expand if needed
# classic martini variants, e.g. "Dry Gin Martini", "Dirty Vodka Martini", etc.
MARTINI_VARIANTS_RE = re.compile(
    r"(?:(?:\b(dry|dirty|wet|filthy)\b.*\bmartini\b)|(?:\b(gin|vodka)\s+martini\b)|(?:\bmartini\b.*\b(dry|dirty|wet|filthy)\b))",
    flags=re.IGNORECASE
)

def is_shaken(method: str) -> bool:
    return isinstance(method, str) and ("shake" in method.lower())

def is_manhattan(name: str) -> bool:
    return isinstance(name, str) and ("manhattan" in name.lower())

def is_classic_martini(name: str) -> bool:
    if not isinstance(name, str):
        return False
    n = name.lower()
    if any(bad in n for bad in MARTINI_EXCLUDE):
        return False
    return bool(MARTINI_VARIANTS_RE.search(n))

def is_vesper(name: str) -> bool:
    return isinstance(name, str) and ("vesper" in name.lower())

def _passes_constraints(df_choice: pd.DataFrame, mode_key: str) -> bool:
    # Global: no more than 2 shaken
    shaken_count = sum(is_shaken(m) for m in df_choice["Method"].astype(str))
    if shaken_count > 2:
        return False

    # Only enforce the name-family constraints for non-menu rounds
    if mode_key != "menu":
        names = df_choice["Drink"].astype(str).tolist()
        manhattan_count = sum(is_manhattan(n) for n in names)
        if manhattan_count > 1:
            return False

        martini_count = sum(is_classic_martini(n) for n in names)
        vesper_count = sum(is_vesper(n) for n in names)

        # at most one from the union of {classic martini variants} ∪ {vesper}
        if (martini_count + vesper_count) > 1:
            return False

    return True

def sample_by_mix(df: pd.DataFrame, chosen_rank_key: str, max_tries: int = 2000) -> pd.DataFrame:
    """
    Pick drinks per MIX_RULES for the chosen rank, subject to constraints:
      - ≤2 shaken across the whole round (all modes)
      - (non-menu modes) ≤1 'manhattan'
      - (non-menu modes) ≤1 from {classic martini variants, vesper}
    Tries up to max_tries random combinations; raises with a clear message if infeasible.
    """
    if chosen_rank_key not in MIX_RULES:
        raise ValueError(f"Unknown rank: {chosen_rank_key}")

    # Precompute pools by bucket
    base = df.dropna(subset=["Drink"]).copy()
    base["RankKey"] = base["Rank"].astype(str).strip().str.lower()

    required = MIX_RULES[chosen_rank_key]
    pools = {}
    for bucket, need in required.items():
        pool = base[base["RankKey"] == bucket]
        if len(pool) < need:
            raise ValueError(f"Not enough drinks for bucket '{bucket.title()}': need {need}, have {len(pool)}.")
        pools[bucket] = pool

    # Fast reject if total feasible shaken limit is impossible (optional heuristic)
    # e.g., if every item in every pool is shaken and total needed > 2
    total_needed = sum(required.values())
    total_min_non_shake = sum(max(0, len(pools[b]) - sum(is_shaken(m) for m in pools[b]["Method"].astype(str))) for b in pools)
    # This heuristic is conservative; we rely on randomized search anyway.

    # Randomized search with bounded retries (small rounds → fast)
    for _ in range(max_tries):
        picks = []
        for bucket, need in required.items():
            picks.append(pools[bucket].sample(n=need, replace=False))
        choice = pd.concat(picks, ignore_index=True)

        if _passes_constraints(choice, chosen_rank_key):
            return choice

    # If we get here, likely infeasible with current constraints/data
    # Provide diagnostics
    diag = []
    # Count shaken availability
    for bucket, pool in pools.items():
        shaken_in_pool = sum(is_shaken(m) for m in pool["Method"].astype(str))
        diag.append(f"{bucket.title()}: {len(pool)} items ({shaken_in_pool} shaken) need {required[bucket]}")
    raise ValueError(
        "Could not find a combination that satisfies all constraints after "
        f"{max_tries} tries.\n"
        "Tips:\n"
        "- Reduce the number of shaken drinks in your pools or relax the ≤2 shaken rule.\n"
        "- Ensure there aren't too many 'Manhattan' or classic martini/vesper entries per mode.\n"
        "- Details:\n  " + "\n  ".join(diag)
    )


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Round Builder", layout="centered")
st.title("🍸 Round Builder (uses local Quiz Prep.xlsx)")

# Load the workbook once
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

# Sidebar controls
st.sidebar.markdown("**Mode**")
mode = st.sidebar.radio("Choose mode", ["Rank mix", "Custom (pick 4–6 drinks)"], index=0)

# Shared action button
generate = st.sidebar.button("Generate round")

# Decide how to get the working set
sampled_rows = None

if mode == "Rank mix":
    # Rank mode like before
    rank_choice = st.sidebar.selectbox("Choose rank", ["Menu","Certified","Junior","Senior"])
    if generate:
        chosen_key = rank_choice.lower()
        try:
            sampled_rows = sample_by_mix(df, chosen_key)
        except Exception as e:
            st.error(str(e))
            st.stop()

else:
    # Custom selection mode
    st.sidebar.markdown("**Custom selection**")
    # Optional filter by rank to shorten the list
    rank_filter = st.sidebar.multiselect(
        "Filter by rank (optional)",
        ["Menu","Certified","Junior","Senior"],
        default=["Menu","Certified","Junior","Senior"]
    )
    rank_keys = {r.lower() for r in rank_filter}
    df["RankKey"] = df["Rank"].astype(str).str.strip().str.lower()
    df_filtered = df[df["RankKey"].isin(rank_keys)] if rank_filter else df

    # Multiselect drinks
    all_drinks = sorted(df_filtered["Drink"].dropna().astype(str).unique())
    chosen_drinks = st.sidebar.multiselect(
        "Select 4–6 drinks",
        options=all_drinks
    )

    if generate:
        if not (4 <= len(chosen_drinks) <= 6):
            st.error("Please select between 4 and 6 drinks.")
            st.stop()
        sampled_rows = df[df["Drink"].astype(str).isin(chosen_drinks)].copy()

# ---------- Build + display results ----------
if sampled_rows is not None and not sampled_rows.empty:

    # ---------- Ingredient parsing ----------
    drink_to_ingredients_set = {
        row.Drink: parse_ingredients_cell_set(row.Ingredients)
        for row in sampled_rows.itertuples(index=False)
    }
    drink_to_ingredients_list = {
        row.Drink: parse_ingredients_cell_list(row.Ingredients)
        for row in sampled_rows.itertuples(index=False)
    }

    # ---------- Round overview ----------
    st.subheader("Your round")
    for i, row in enumerate(sampled_rows.itertuples(index=False), start=1):
        st.write(f"**{i}. {row.Drink}** — {row.Ingredients}")
    # ---------- Mat order ----------
    sampled_rows = sampled_rows.assign(
        MethodPriority=sampled_rows["Method"].apply(get_method_priority),
        MethodComplexity=sampled_rows["Method"].apply(get_method_complexity),
    )

    base_sorted = sampled_rows.sort_values(
        by=["MethodPriority", "MethodComplexity", "Drink"]
    ).reset_index(drop=True)

    final_chunks = []
    for _, group_df in base_sorted.groupby("MethodPriority", sort=True):
        clustered = cluster_by_ingredients(group_df, drink_to_ingredients_set)
        final_chunks.append(clustered)

    final_mat_order_df = pd.concat(final_chunks, ignore_index=True)

    st.subheader("Suggested mat order (Left → Right)")
    for i, row in enumerate(final_mat_order_df.itertuples(index=False), start=1):
        st.write(f"{i}. **{row.Drink}** — {row.Method}")
        
    # ---------- Common ingredients ----------
    ingredient_to_drinks = {}
    for drink, ings in drink_to_ingredients_set.items():
        for ing in ings:
            ingredient_to_drinks.setdefault(ing, []).append(drink)
    common_ingredients = {
        ing: drinks for ing, drinks in ingredient_to_drinks.items() if len(drinks) >= 2
    }

    st.subheader("Common ingredients")
    if common_ingredients:
        for ing, drinks in sorted(common_ingredients.items(),
                                  key=lambda kv: (-len(kv[1]), kv[0])):
            st.write(f"- **{ing.capitalize()}** appears in: {', '.join(drinks)}")
    else:
        st.write("No common ingredients found between these drinks.")
    st.write("")

    # ---------- Order of operations ----------
    st.subheader("Order of operations per drink")
    for row in final_mat_order_df.itertuples(index=False):
        ing_list   = drink_to_ingredients_list.get(row.Drink, [])
        method_str = getattr(row, "Method", "") or ""
        glass_str  = getattr(row, "Glass", "") or "Glassware N/A"
        ice_str    = getattr(row, "Ice", "") or "Ice N/A"
        garnish_str= getattr(row, "Garnish", "") or "Garnish N/A"

        misc, non_alc, alc, toppers = split_for_order_of_ops(ing_list, method_str)

        st.markdown(
            f"**{row.Drink}** — Glass: _{glass_str}_ — Ice: _{ice_str}_ — "
            f"Garnish: _{garnish_str}_ — Method: _{method_str}_"
        )

        def pretty(items): return ", ".join(items) if items else "—"
        st.write(f"- Misc: {pretty(misc)}")
        st.write(f"- Non-alcoholic: {pretty(non_alc)}")
        st.write(f"- Alcoholic: {pretty(alc)}")
        if toppers:
            st.write(f"- Top (last): {pretty(toppers)}")

    st.success("Round generated. ✅")

else:
    st.info("Choose a mode, select options, then click **Generate round**.")

