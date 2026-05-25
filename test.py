"""
Fantrax Free Agent Z-Score Finder — KC Sandlot Points League
=============================================================
Uses two public Fantrax endpoints (no auth required):

  1. getLeagueInfo  → playerInfo: fantraxId → {eligiblePos, status}
       https://www.fantrax.com/fxea/general/getLeagueInfo?leagueId=<ID>

  2. getPlayerIds   → fantraxId → {name, team, position, ...}
       https://www.fantrax.com/fxea/general/getPlayerIds?sport=MLB
       (discovered via github.com/pmurley/go-fantrax)

Player name → MLBAM ID bridge:
  getPlayerIds names are joined to MLB Stats API data via the Chadwick Bureau
  register (pybaseball.chadwick_register), which maps player names to MLBAM
  integer IDs. The final join is on integer MLBAM ID — no fuzzy name matching —
  so stale or wrong names in getPlayerIds produce a failed lookup (player
  silently dropped) rather than a wrong-player attribution.

Scoring system (points league — exact weights from league settings):
  HITTING:  R+1, RBI+1, SB+2, BB+1, SO-0.5, HBP+1, TB+1
  PITCHING: W+5, L-2, SV+5, BS-2, HLD+3, IP+3, H-1, ER-1,
            BB-1, K+0.5, HB-1, QS+3, PKO+1

Prerequisites:
  pip install mlb-statsapi pandas requests pybaseball

Usage:
  python fantrax_free_agent_zscores.py --league LEAGUE_ID [options]

Options:
  --league ID       Fantrax league ID (or set FANTRAX_LEAGUE_ID env var)
  --season YEAR     MLB season year (default: 2026)
  --min-pa INT      Min plate appearances to include a hitter (default: 5)
  --min-ip FLOAT    Min innings pitched to include a pitcher (default: 1)
  --position POS    Filter by position: C 1B 2B 3B SS OF SP RP UT
  --status STR      FA | WW | A=all available (default: A)
  --days-h INT      Hitting window in days (default: 14)
  --out FILE        Output CSV path (default: free_agents.csv)
  --top INT         Rows per section in console + CSV (default: 20)

CSV layout (4 sections, 2 blank rows between):
  Hitters  — Last 7 days
  Hitters  — Last N days  (--days-h)
  Pitchers — Last 14 days
  Pitchers — Last 30 days
"""

import argparse
import os
import re
import warnings
from datetime import date, timedelta

import pandas as pd
import pybaseball
import requests
import statsapi

warnings.filterwarnings("ignore")   # suppress pybaseball download messages

# ── Fantrax endpoints ─────────────────────────────────────────────────────────
FANTRAX_BASE         = "https://www.fantrax.com/fxea/general"
LEAGUE_INFO_URL      = f"{FANTRAX_BASE}/getLeagueInfo"
PLAYER_IDS_URL       = f"{FANTRAX_BASE}/getPlayerIds"

FANTRAX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":  "application/json, text/plain, */*",
    "Referer": "https://www.fantrax.com/",
}

PITCHER_POS = {"SP", "RP"}

# ── Scoring system ────────────────────────────────────────────────────────────
# Maps MLB Stats API field name → fantasy points per unit
# Hitting fields (byDateRange, group=hitting):
HITTING_SCORING: dict[str, float] = {
    "runs":        1.0,   # R  — Runs Scored
    "rbi":         1.0,   # RBI
    "stolenBases": 2.0,   # SB — Stolen Bases
    "baseOnBalls": 1.0,   # BB — Walks
    "strikeOuts": -0.5,   # SO — Batter strikeouts (NEGATIVE)
    "hitByPitch":  1.0,   # HBP — Hit By Pitches
    "totalBases":  1.0,   # TB — Total Bases
}

# Pitching fields (byDateRange, group=pitching):
PITCHING_SCORING: dict[str, float] = {
    "wins":          5.0,   # W
    "losses":       -2.0,   # L
    "saves":         5.0,   # SV
    "blownSaves":   -2.0,   # BS — Blown Saves
    "holds":         3.0,   # HLD
    "inningsPitched": 3.0,  # IP — parsed as true decimal innings
    "hits":         -1.0,   # H  — Hits Allowed
    "earnedRuns":   -1.0,   # ER — Earned Runs Allowed
    "baseOnBalls":  -1.0,   # BB — Walks Allowed (NEGATIVE)
    "strikeOuts":    0.5,   # K  — Pitcher strikeouts
    "hitBatsmen":   -1.0,   # HB — Hit Batsmen (NEGATIVE)
    "qualityStarts": 3.0,   # QS — Quality Starts
    "pickoffs":      1.0,   # PKO — Pickoffs
}

# Additional raw fields to fetch but not score (used for PA/IP filters + display)
HITTING_EXTRA  = ["plateAppearances"]
PITCHING_EXTRA = ["era", "whip"]   # kept for display reference only


# ── Fantrax data fetch ────────────────────────────────────────────────────────

def fetch_league_info(league_id: str) -> dict:
    print(f"⏬  getLeagueInfo  {league_id} …")
    r = requests.get(LEAGUE_INFO_URL, params={"leagueId": league_id},
                     headers=FANTRAX_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_player_id_map() -> dict[str, dict]:
    """
    GET /fxea/general/getPlayerIds?sport=MLB
    Returns { fantraxId: {name, team, position, fantraxId, statsIncId?, ...} }
    Team-aggregate rows (empty 'team') are filtered out per go-fantrax logic.
    """
    print("⏬  getPlayerIds   sport=MLB …")
    r = requests.get(PLAYER_IDS_URL, params={"sport": "MLB"},
                     headers=FANTRAX_HEADERS, timeout=60)
    r.raise_for_status()
    raw: dict = r.json()
    players = {fid: info for fid, info in raw.items() if info.get("team", "") != ""}
    print(f"   → {len(players):,} individual players in Fantrax MLB database")
    return players


def parse_available_players(
    league_info: dict,
    status_filter: str = "A",
    position_filter: str | None = None,
) -> dict[str, dict]:
    """Extract available player IDs + positions from playerInfo."""
    player_info: dict = league_info.get("playerInfo", {})
    sf = status_filter.upper()
    wanted = {"FA"} if sf == "FA" else {"WW"} if sf in ("W", "WW") else {"FA", "WW"}

    out: dict[str, dict] = {}
    for fid, info in player_info.items():
        if info["status"] not in wanted:
            continue
        pos_str  = info.get("eligiblePos", "")
        pos_list = [p.strip() for p in pos_str.split(",") if p.strip()]
        if position_filter and position_filter.upper() not in {p.upper() for p in pos_list}:
            continue
        out[fid] = {"eligiblePos": pos_str, "positions": pos_list, "status": info["status"]}

    print(f"   → {len(out):,} available players (status={status_filter})")
    return out


def build_fa_list(
    available: dict[str, dict],
    player_id_map: dict[str, dict],
) -> list[dict]:
    """Join available IDs (getLeagueInfo) with names (getPlayerIds) on fantraxId."""
    players, unresolved = [], 0
    for fid, avail in available.items():
        info = player_id_map.get(fid)
        if info is None:
            unresolved += 1
            continue
        players.append({
            "id":          fid,
            "name":        info.get("name", ""),
            "team":        info.get("team", ""),
            "eligiblePos": avail["eligiblePos"],
            "positions":   avail["positions"],
            "status":      avail["status"],
        })
    print(f"   → {len(players):,}/{len(available):,} FA IDs resolved to names "
          f"({unresolved} unmatched in player map)")
    return players


def split_by_position(fa_players: list[dict]) -> tuple[list, list]:
    hitters, pitchers = [], []
    for p in fa_players:
        (pitchers if set(p["positions"]) & PITCHER_POS else hitters).append(p)
    return hitters, pitchers


# ── MLB Stats API fetch + fantasy point computation ───────────────────────────

def _date_range(days: int) -> tuple[str, str]:
    end   = date.today()
    start = end - timedelta(days=days)
    fmt   = "%Y-%m-%d"
    return start.strftime(fmt), end.strftime(fmt)


def _parse_ip(ip_str: str) -> float:
    """Convert Fantrax/MLB fractional IP string to true decimal innings.
    '6.2' means 6 and 2/3 innings = 6.667, NOT 6.2."""
    parts = str(ip_str).split(".")
    full  = int(parts[0])
    outs  = int(parts[1]) if len(parts) == 2 else 0
    return full + outs / 3


def _safe_float(val) -> float:
    try:
        return float(val) if val else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(val) if val else 0
    except (TypeError, ValueError):
        return 0


def _compute_hitting_pts(st: dict) -> float:
    """Compute fantasy points from a hitting stat dict using league scoring."""
    pts = 0.0
    for field, weight in HITTING_SCORING.items():
        pts += _safe_float(st.get(field, 0)) * weight
    return round(pts, 2)


def _compute_pitching_pts(st: dict) -> float:
    """Compute fantasy points from a pitching stat dict using league scoring.
    IP is treated as fractional outs (e.g. '6.2' = 6⅔ IP = 6.667 true innings).
    """
    pts = 0.0
    for field, weight in PITCHING_SCORING.items():
        if field == "inningsPitched":
            pts += _parse_ip(st.get("inningsPitched", "0")) * weight
        else:
            pts += _safe_float(st.get(field, 0)) * weight
    return round(pts, 2)


def fetch_hitting_stats(season: int, days: int) -> pd.DataFrame:
    start, end = _date_range(days)
    print(f"⏬  MLB hitting     last {days:2d} days  ({start} → {end}) …")
    data   = statsapi.get("stats", {
        "stats": "byDateRange", "season": season,
        "group": "hitting", "sportId": 1, "limit": 2000,
        "startDate": start, "endDate": end,
    })
    rows = []
    for s in data["stats"][0]["splits"]:
        p, st = s["player"], s["stat"]
        row = {
            "playerId":          p["id"],
            "name":              p["fullName"],
            "pa":                _safe_int(st.get("plateAppearances", 0)),
            # Scoring stats
            "runs":              _safe_int(st.get("runs", 0)),
            "rbi":               _safe_int(st.get("rbi", 0)),
            "stolenBases":       _safe_int(st.get("stolenBases", 0)),
            "baseOnBalls":       _safe_int(st.get("baseOnBalls", 0)),
            "strikeOuts":        _safe_int(st.get("strikeOuts", 0)),
            "hitByPitch":        _safe_int(st.get("hitByPitch", 0)),
            "totalBases":        _safe_int(st.get("totalBases", 0)),
        }
        row["fpts"] = _compute_hitting_pts(st)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"   → {len(df):,} hitters")
    return df


def fetch_pitching_stats(season: int, days: int) -> pd.DataFrame:
    start, end = _date_range(days)
    print(f"⏬  MLB pitching    last {days:2d} days  ({start} → {end}) …")
    data   = statsapi.get("stats", {
        "stats": "byDateRange", "season": season,
        "group": "pitching", "sportId": 1, "limit": 2000,
        "startDate": start, "endDate": end,
    })
    rows = []
    for s in data["stats"][0]["splits"]:
        p, st = s["player"], s["stat"]
        ip = _parse_ip(st.get("inningsPitched", "0"))
        row = {
            "playerId":       p["id"],
            "name":           p["fullName"],
            "ip":             ip,
            # Scoring stats (raw)
            "wins":           _safe_int(st.get("wins", 0)),
            "losses":         _safe_int(st.get("losses", 0)),
            "saves":          _safe_int(st.get("saves", 0)),
            "blownSaves":     _safe_int(st.get("blownSaves", 0)),
            "holds":          _safe_int(st.get("holds", 0)),
            "hits":           _safe_int(st.get("hits", 0)),
            "earnedRuns":     _safe_int(st.get("earnedRuns", 0)),
            "baseOnBalls":    _safe_int(st.get("baseOnBalls", 0)),
            "strikeOuts":     _safe_int(st.get("strikeOuts", 0)),
            "hitBatsmen":     _safe_int(st.get("hitBatsmen", 0)),
            "qualityStarts":  _safe_int(st.get("qualityStarts", 0)),
            "pickoffs":       _safe_int(st.get("pickoffs", 0)),
            # Display-only
            "era":            _safe_float(st.get("era")),
            "whip":           _safe_float(st.get("whip")),
        }
        row["fpts"] = _compute_pitching_pts(st)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"   → {len(df):,} pitchers")
    return df


def add_zscores(df: pd.DataFrame, scoring: dict[str, float]) -> pd.DataFrame:
    """
    For each scored stat column, compute a z-score and accumulate into
    'total-zscore'.  Negative-weight stats are inverted so that a lower
    raw value (e.g. fewer strikeouts, fewer earned runs) contributes
    positively to the total.

    Sort order: FPts descending, then total-zscore descending.
    """
    df = df.copy()
    z_cols: list[str] = []

    for field, weight in scoring.items():
        col = field if field != "inningsPitched" else "ip"
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std(ddof=0)
        if sigma == 0:
            df[f"_z_{field}"] = 0.0
        else:
            raw_z = (df[col] - mu) / sigma
            # Invert so that "less is better" stats still add positively
            df[f"_z_{field}"] = -raw_z if weight < 0 else raw_z
        z_cols.append(f"_z_{field}")

    df["total-zscore"] = df[z_cols].sum(axis=1).round(2)
    df = df.drop(columns=z_cols)

    return df.sort_values(
        ["fpts", "total-zscore"], ascending=[False, False]
    ).reset_index(drop=True)


# ── Chadwick register: name → MLBAM ID bridge ────────────────────────────────

_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", re.IGNORECASE)

def _chadwick_name(name: str) -> str:
    """Normalize a player name for Chadwick lookup.
    Strips Jr/Sr/II/III suffixes that Chadwick omits from its name fields.
    """
    return _SUFFIX_RE.sub("", name.strip().lower())


def build_chadwick_lookup() -> dict[str, int]:
    """
    Download the Chadwick Bureau register (cached locally after first run)
    and return a dict mapping normalized full name → MLBAM integer ID.
    Where duplicate names exist, the player with the most recent MLB activity wins.
    """
    print("⏬  Chadwick register  (name → MLBAM ID) …")
    register = pybaseball.chadwick_register()
    register = register[register["key_mlbam"].notna() & (register["key_mlbam"] > 0)].copy()
    register["key_mlbam"] = register["key_mlbam"].astype(int)
    register["_name"] = (
        register["name_first"].fillna("") + " " + register["name_last"].fillna("")
    ).str.strip().str.lower()

    # Most recent player wins on duplicate names
    register = register.sort_values("mlb_played_last", ascending=False, na_position="last")
    register = register.drop_duplicates(subset="_name", keep="first")
    lookup = dict(zip(register["_name"], register["key_mlbam"]))
    print(f"   → {len(lookup):,} name→MLBAM mappings loaded")
    return lookup


def resolve_mlbam_ids(
    fa_players: list[dict],
    chadwick: dict[str, int],
) -> list[dict]:
    """
    Add 'mlbam_id' to each FA player dict by looking up their name in the
    Chadwick register.  Players whose name cannot be resolved are kept with
    mlbam_id=None and will be silently dropped at join time.
    """
    resolved = unresolved = 0

    # Diagnostic: show a few raw Fantrax names vs what Chadwick expects
    sample_names = [p.get("name", "") for p in fa_players[:10] if p.get("name")]
    if sample_names:
        print(f"   sample Fantrax names (raw):     {sample_names[:5]}")
        print(f"   sample Chadwick names (sample): {list(chadwick.keys())[:5]}")

    for p in fa_players:
        norm = _chadwick_name(p.get("name", ""))
        mlbam = chadwick.get(norm)
        p["mlbam_id"] = mlbam
        if mlbam:
            resolved += 1
        else:
            unresolved += 1
    print(f"   → {resolved} FA names resolved to MLBAM ID, {unresolved} unresolved (skipped)")
    return fa_players


def join_fa_to_stats(
    fa_players: list[dict],
    zscore_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner join FA players to the z-scored MLB stats DataFrame on integer MLBAM ID.
    No name matching — clean integer join eliminates wrong-player attributions.
    """
    if not fa_players or zscore_df.empty:
        return pd.DataFrame()

    resolved = [p for p in fa_players if p.get("mlbam_id")]
    if not resolved:
        return pd.DataFrame()

    fa_df = pd.DataFrame([
        {
            "mlbam_id":     p["mlbam_id"],
            "fantrax_id":   p["id"],
            "fantrax_name": p["name"],
            "fantrax_team": p.get("team", ""),
            "eligiblePos":  p.get("eligiblePos", ""),
        }
        for p in resolved
    ])

    merged = fa_df.merge(
        zscore_df,
        left_on="mlbam_id",
        right_on="playerId",
        how="inner",
    )
    return merged.sort_values(
        ["fpts", "total-zscore"], ascending=[False, False]
    ).reset_index(drop=True)


# ── Column specs ──────────────────────────────────────────────────────────────

HITTER_COLS: list[tuple[str, str]] = [
    ("fantrax_name",  "Name"),
    ("fantrax_team",  "Team"),
    ("eligiblePos",   "Pos"),
    ("pa",            "PA"),
    ("fpts",          "FPts"),
    ("total-zscore",  "total-zscore"),
]

PITCHER_COLS: list[tuple[str, str]] = [
    ("fantrax_name",  "Name"),
    ("fantrax_team",  "Team"),
    ("eligiblePos",   "Pos"),
    ("ip",            "IP"),
    ("fpts",          "FPts"),
    ("total-zscore",  "total-zscore"),
]

HIT_ROUND = {"FPts": 1, "IP": 1}
PIT_ROUND = {"FPts": 1, "IP": 1}


def _build_table(df: pd.DataFrame, cols: list[tuple], top: int,
                 rounding: dict) -> pd.DataFrame:
    src   = [s for s, _ in cols if s in df.columns]
    names = [d for s, d in cols if s in df.columns]
    out   = df[src].head(top).copy()
    out.columns = names
    for col, places in rounding.items():
        if col in out.columns:
            out[col] = out[col].round(places)
    return out


# ── CSV output ────────────────────────────────────────────────────────────────

def write_combined_csv(sections: list[tuple], path: str, top: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        for i, (label, df, cols, rounding) in enumerate(sections):
            if i > 0:
                f.write("\n\n")
            f.write(f"{label}\n")
            if df.empty:
                f.write("No data\n")
            else:
                _build_table(df, cols, top, rounding).to_csv(f, index=False)
    print(f"✅  {path} written")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fantrax Free Agent Fantasy-Point Z-Score Finder"
    )
    ap.add_argument("--league",   default=os.environ.get("FANTRAX_LEAGUE_ID"),
                    help="Fantrax league ID (or set FANTRAX_LEAGUE_ID env var)")
    ap.add_argument("--season",   type=int,   default=2026)
    ap.add_argument("--min-pa",   type=int,   default=5,  dest="min_pa",
                    help="Min PA to include a hitter in z-score pool (default: 5)")
    ap.add_argument("--min-ip",   type=float, default=1,  dest="min_ip",
                    help="Min IP to include a pitcher in z-score pool (default: 1)")
    ap.add_argument("--position", default=None,
                    help="Position filter: C 1B 2B 3B SS OF SP RP UT")
    ap.add_argument("--status",   default="A",
                    help="FA | WW | A=all available (default: A)")
    ap.add_argument("--days-h",   type=int, default=14, dest="days_h",
                    help="Hitting lookback window in days (default: 14)")
    ap.add_argument("--days-p",   type=int, default=14, dest="days_p",
                    help="Pitching lookback window in days (default: 14)")
    ap.add_argument("--out",      default="free_agents.csv")
    ap.add_argument("--top",      type=int, default=20)
    args = ap.parse_args()

    if not args.league:
        ap.error(
            "League ID required. Pass --league YOUR_LEAGUE_ID "
            "or set FANTRAX_LEAGUE_ID env var.\n"
            "  Your ID is in the Fantrax URL: "
            "fantrax.com/fantasy/league/<LEAGUE_ID>/..."
        )

    # ── 1. Fantrax: get available player IDs ──────────────────────────────────
    league_info = fetch_league_info(args.league)
    print(f"✅  League: {league_info.get('leagueName', args.league)}")

    available = parse_available_players(
        league_info, status_filter=args.status, position_filter=args.position
    )

    # ── 2. Fantrax: get player names ──────────────────────────────────────────
    player_id_map = fetch_player_id_map()

    # ── 3. Join: ID + eligibility + name ─────────────────────────────────────
    fa_players = build_fa_list(available, player_id_map)
    fa_hitters, fa_pitchers = split_by_position(fa_players)
    print(f"   → {len(fa_hitters):,} FA hitters / {len(fa_pitchers):,} FA pitchers")

    # ── 4. Chadwick: resolve Fantrax names → MLBAM integer IDs ───────────────
    chadwick = build_chadwick_lookup()
    fa_hitters  = resolve_mlbam_ids(fa_hitters,  chadwick)
    fa_pitchers = resolve_mlbam_ids(fa_pitchers, chadwick)

    # ── 5. MLB Stats API: fetch stats + compute fantasy points + z-score ──────
    hit7_df  = fetch_hitting_stats(args.season, 7)
    hit7_df  = hit7_df[hit7_df["pa"] >= args.min_pa].reset_index(drop=True)
    hit7_df  = add_zscores(hit7_df, HITTING_SCORING)

    hitN_df  = fetch_hitting_stats(args.season, args.days_h)
    hitN_df  = hitN_df[hitN_df["pa"] >= args.min_pa].reset_index(drop=True)
    hitN_df  = add_zscores(hitN_df, HITTING_SCORING)

    pit14_df = fetch_pitching_stats(args.season, 14)
    pit14_df = pit14_df[pit14_df["ip"] >= args.min_ip].reset_index(drop=True)
    pit14_df = add_zscores(pit14_df, PITCHING_SCORING)

    pit30_df = fetch_pitching_stats(args.season, 30)
    pit30_df = pit30_df[pit30_df["ip"] >= args.min_ip].reset_index(drop=True)
    pit30_df = add_zscores(pit30_df, PITCHING_SCORING)

    # ── 6. Join FA list to stats on MLBAM ID ─────────────────────────────────
    def matched(fa_list: list[dict], zscore_df: pd.DataFrame) -> pd.DataFrame:
        return join_fa_to_stats(fa_list, zscore_df)

    hit7_fa  = matched(fa_hitters,  hit7_df)
    hitN_fa  = matched(fa_hitters,  hitN_df)
    pit14_fa = matched(fa_pitchers, pit14_df)
    pit30_fa = matched(fa_pitchers, pit30_df)

    h7_label = "HITTERS — Last 7 Days"
    hN_label = f"HITTERS — Last {args.days_h} Days"
    p14_label = "PITCHERS — Last 14 Days"
    p30_label = "PITCHERS — Last 30 Days"

    sections = [
        (h7_label,  hit7_fa,  HITTER_COLS,  HIT_ROUND),
        (hN_label,  hitN_fa,  HITTER_COLS,  HIT_ROUND),
        (p14_label, pit14_fa, PITCHER_COLS, PIT_ROUND),
        (p30_label, pit30_fa, PITCHER_COLS, PIT_ROUND),
    ]

    # ── 6. Console summary ────────────────────────────────────────────────────
    scoring_note = (
        "Scoring: R+1 RBI+1 SB+2 BB+1 SO-0.5 HBP+1 TB+1 | "
        "W+5 L-2 SV+5 BS-2 HLD+3 IP+3 H-1 ER-1 BB-1 K+0.5 HB-1 QS+3 PKO+1"
    )
    print(f"\n{scoring_note}")

    for label, df, cols, rounding in sections:
        print(f"\n{'═'*76}")
        print(f"  {label}")
        print(f"{'═'*76}")
        if df.empty:
            print("  No matches — try --min-pa 1 or --min-ip 0.1")
        else:
            tbl = _build_table(df, cols, args.top, rounding)
            print(tbl.to_string(index=False))

    # ── 7. Write CSV ──────────────────────────────────────────────────────────
    print()
    write_combined_csv(sections, args.out, args.top)


if __name__ == "__main__":
    main()
