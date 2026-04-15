import os
import json
import re
import sys
import requests
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
NTFY_TOPIC = os.environ["NTFY_TOPIC"]
SYMBOL = "BTCUSDT"
LIMIT = 250

# ATR params
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0
ATR_TRAIL_MULT = 1.5
TRAIL_ACTIVATE_PCT = 0.015
TAKE_PROFIT_MULT = 4.0

# Stop-loss / take-profit safety bounds
STOP_MIN_PCT = 0.01
STOP_MAX_PCT = 0.04
TARGET_MIN_PCT = 0.02
TARGET_MAX_PCT = 0.06


# ─────────────────────────────────────────
# 1. Klines (CryptoCompare)
# ─────────────────────────────────────────
def get_klines():
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    r = requests.get(url, params={"fsym": "BTC", "tsym": "USDT", "limit": LIMIT}, timeout=10)
    r.raise_for_status()
    data = r.json()["Data"]["Data"]
    df = pd.DataFrame(data)
    df = df.rename(columns={"time": "open_time", "volumefrom": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


# ─────────────────────────────────────────
# 2. Derivatives sentiment (Bybit, optional)
# ─────────────────────────────────────────
def get_funding_rate():
    try:
        r = requests.get("https://api.bybit.com/v5/market/tickers",
                         params={"category": "linear", "symbol": SYMBOL}, timeout=10)
        r.raise_for_status()
        data = r.json()["result"]["list"]
        if data:
            return float(data[0]["fundingRate"])
    except Exception as e:
        print(f"Funding rate fetch failed: {e}")
    return None


def get_long_short_ratio():
    try:
        r = requests.get("https://api.bybit.com/v5/market/account-ratio",
                         params={"category": "linear", "symbol": SYMBOL,
                                 "period": "1h", "limit": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()["result"]["list"]
        if data:
            return float(data[0]["buyRatio"]) / float(data[0]["sellRatio"])
    except Exception as e:
        print(f"Long/short ratio fetch failed: {e}")
    return None


def get_open_interest_change():
    try:
        r = requests.get("https://api.bybit.com/v5/market/open-interest",
                         params={"category": "linear", "symbol": SYMBOL,
                                 "intervalTime": "1h", "limit": 5}, timeout=10)
        r.raise_for_status()
        data = r.json()["result"]["list"]
        if len(data) >= 2:
            latest = float(data[0]["openInterest"])
            oldest = float(data[-1]["openInterest"])
            price_r = requests.get("https://api.bybit.com/v5/market/tickers",
                                   params={"category": "linear", "symbol": SYMBOL}, timeout=10)
            price_r.raise_for_status()
            price = float(price_r.json()["result"]["list"][0]["lastPrice"])
            latest_val = latest * price
            oldest_val = oldest * price
            change_pct = round((latest_val - oldest_val) / oldest_val * 100, 2)
            return change_pct, round(latest_val / 1e8, 2)
    except Exception as e:
        print(f"Open interest fetch failed: {e}")
    return None, None


def get_fear_greed():
    """Crypto Fear & Greed Index (0-100). Free, no API key, no IP restriction."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        data = r.json()["data"][0]
        value = int(data["value"])
        label = data["value_classification"]  # e.g. "Fear", "Greed", "Extreme Fear"
        return value, label
    except Exception as e:
        print(f"Fear & Greed fetch failed: {e}")
        return None, None


# ─────────────────────────────────────────
# 3. Technical indicators
# ─────────────────────────────────────────
def compute_indicators(df):
    df["ma200"] = df["close"].rolling(200).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["recent_high_40"] = df["high"].rolling(40).max()
    df["recent_low_40"] = df["low"].rolling(40).min()

    df["prev_close"] = df["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_close"]).abs(),
        (df["low"] - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()

    return df


# ─────────────────────────────────────────
# 4. Dynamic stop-loss / take-profit
# ─────────────────────────────────────────
def calc_dynamic_levels(entry_price, atr):
    raw_stop = entry_price - atr * ATR_STOP_MULT
    raw_target = entry_price + atr * TAKE_PROFIT_MULT

    stop = max(raw_stop, entry_price * (1 - STOP_MAX_PCT))
    stop = min(stop, entry_price * (1 - STOP_MIN_PCT))
    target = max(raw_target, entry_price * (1 + TARGET_MIN_PCT))
    target = min(target, entry_price * (1 + TARGET_MAX_PCT))

    stop_pct = (stop / entry_price - 1) * 100
    target_pct = (target / entry_price - 1) * 100
    return round(stop, 1), round(target, 1), round(stop_pct, 2), round(target_pct, 2)


def calc_trailing_stop(highest_price, atr):
    return round(highest_price - atr * ATR_TRAIL_MULT, 1)


# ─────────────────────────────────────────
# 5. Signal scoring
# ─────────────────────────────────────────
SCORE_LABELS = {
    "Trend MA200": None,
    "MA50 Slope": None,
    "RSI Momentum": None,
    "Volume": None,
    "Position": None,
    "Support": None,
}


def check_signal(df, funding_rate, ls_ratio):
    last = df.iloc[-1]
    prev3 = df.iloc[-4]
    price = last["close"]
    ma200 = last["ma200"]
    ma50 = last["ma50"]
    rsi_now = last["rsi"]
    rsi_3ago = prev3["rsi"]
    vol_now = last["volume"]
    vol_ma = last["vol_ma20"]
    recent_high = last["recent_high_40"]
    recent_low = last["recent_low_40"]
    atr = last["atr"]

    score = 0.0
    scores = {}

    # 1. Trend: price above MA200 — weight 3.0
    s = 3.0 if price > ma200 else 0.0
    score += s
    scores["Trend MA200"] = (s, 3.0, price > ma200)

    # 2. MA50 slope — weight 1.5
    ma50_now = last["ma50"]
    ma50_4ago = df.iloc[-4]["ma50"]
    if ma50_now > ma50_4ago:
        s = 1.5
    elif ma50_now > df.iloc[-8]["ma50"]:
        s = 0.5
    else:
        s = 0.0
    score += s
    scores["MA50 Slope"] = (s, 1.5, s > 0)

    # 3. RSI momentum — weight 2.5
    if rsi_3ago < 38 and rsi_now > 44:
        s = 2.5
    elif rsi_3ago < 45 and rsi_now > 44:
        s = 1.0
    elif 50 < rsi_now < 65:
        s = 0.5
    else:
        s = 0.0
    score += s
    scores["RSI Momentum"] = (s, 2.5, s >= 1.0)

    # 4. Volume confirmation — weight 2.0
    vol_ratio = vol_now / vol_ma
    if vol_ratio >= 1.3:
        s = 2.0
    elif vol_ratio >= 1.1:
        s = 1.0
    else:
        s = 0.0
    score += s
    scores["Volume"] = (s, 2.0, s >= 1.0)

    # 5. Not chasing highs — weight 1.5
    dist_from_high = (price / recent_high - 1) * 100
    if dist_from_high < -5:
        s = 1.5
    elif dist_from_high < -3:
        s = 0.8
    else:
        s = 0.0
    score += s
    scores["Position"] = (s, 1.5, s > 0)

    # 6. Near support — weight 0.5
    dist_from_low = (price / recent_low - 1) * 100
    s = 0.5 if dist_from_low < 4 else 0.0
    score += s
    scores["Support"] = (s, 0.5, s > 0)

    score = round(score, 1)

    # Signal strength
    if score >= 9.5:
        strength = "Very Strong"
    elif score >= 8.0:
        strength = "Strong"
    elif score >= 6.5:
        strength = "Moderate"
    else:
        strength = "Weak"

    # Hard filters
    trend_ok = price > ma200

    funding_ok = True
    funding_block_reason = None
    if funding_rate is not None and funding_rate > 0.0008:
        funding_ok = False
        funding_block_reason = f"Funding rate too high ({funding_rate*100:.3f}%), longs crowded"

    ENTRY_THRESHOLD = 6.5
    tech_signal = trend_ok and (score >= ENTRY_THRESHOLD)
    entry_signal = tech_signal and funding_ok
    ls_warning = ls_ratio is not None and ls_ratio > 1.6

    return {
        "price": price,
        "ma200": round(ma200, 1),
        "ma50": round(ma50, 1),
        "rsi": round(rsi_now, 1),
        "rsi_3ago": round(rsi_3ago, 1),
        "vol_ratio": round(vol_ratio, 2),
        "recent_high": round(recent_high, 1),
        "atr": round(atr, 1),
        "atr_pct": round(atr / price * 100, 3),
        "score": score,
        "score_max": 11.0,
        "threshold": ENTRY_THRESHOLD,
        "strength": strength,
        "scores": scores,
        "entry_signal": entry_signal,
        "tech_signal": tech_signal,
        "funding_ok": funding_ok,
        "funding_block_reason": funding_block_reason,
        "ls_warning": ls_warning,
        "trend_broken": price < ma200,
        "ma50_broken": price < ma50,
    }


# ─────────────────────────────────────────
# 6. Position state (JSON)
# ─────────────────────────────────────────
STATE_FILE = "state.json"


def read_state():
    if os.path.exists(STATE_FILE):
        content = open(STATE_FILE).read().strip()
        if content and content != "NONE":
            try:
                return json.loads(content)
            except Exception:
                try:
                    ep = float(content)
                    return {"entry_price": ep, "initial_stop": None,
                            "take_profit": None, "atr": None,
                            "highest": ep, "trailing_stop": None,
                            "trailing_active": False}
                except Exception:
                    pass
    return None


def write_state(state_dict):
    with open(STATE_FILE, "w") as f:
        if state_dict is None:
            f.write("NONE")
        else:
            json.dump(state_dict, f)


# ─────────────────────────────────────────
# 7. Sentiment interpretation
# ─────────────────────────────────────────
def interpret_funding(rate):
    if rate is None:
        return "N/A", "neutral"
    pct = rate * 100
    if pct > 0.08:
        return f"+{pct:.3f}% Overheated", "overheated"
    elif pct > 0.03:
        return f"+{pct:.3f}% Warm", "warm"
    elif pct > -0.01:
        return f"{pct:+.3f}% Neutral", "neutral"
    elif pct > -0.05:
        return f"{pct:+.3f}% Cool", "cool"
    else:
        return f"{pct:+.3f}% Cold", "cold"


def interpret_ls(ratio):
    if ratio is None:
        return "N/A"
    if ratio > 1.5:
        return f"{ratio:.2f} Retail heavily long (contrarian)"
    elif ratio > 1.1:
        return f"{ratio:.2f} Slightly long"
    elif ratio > 0.9:
        return f"{ratio:.2f} Balanced"
    elif ratio > 0.7:
        return f"{ratio:.2f} Slightly short"
    else:
        return f"{ratio:.2f} Retail heavily short (bounce possible)"


def interpret_oi(change_pct):
    if change_pct is None:
        return "N/A"
    if change_pct > 3:
        return f"+{change_pct}% Large inflow"
    elif change_pct > 0:
        return f"+{change_pct}% Small inflow"
    elif change_pct > -3:
        return f"{change_pct}% Small outflow"
    else:
        return f"{change_pct}% Large outflow"


# ─────────────────────────────────────────
# 8. ntfy.sh push notification
# ─────────────────────────────────────────
def send_ntfy(msg, title="BTC Signal"):
    clean = re.sub(r"<[^>]+>", "", msg)
    r = requests.post(
        "https://ntfy.sh/",
        json={"topic": NTFY_TOPIC, "title": title, "message": clean, "priority": 4},
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=10,
    )
    r.raise_for_status()
    print("Notification sent")


# ─────────────────────────────────────────
# 9. Main logic
# ─────────────────────────────────────────
def main():
    now_utc = datetime.utcnow()
    now_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    now_cst_h = (now_utc.hour + 8) % 24

    print(f"[{now_str}] Checking...")

    df = get_klines()
    df = compute_indicators(df)
    funding_rate = get_funding_rate()
    ls_ratio = get_long_short_ratio()
    oi_change, oi_val = get_open_interest_change()
    fng_value, fng_label = get_fear_greed()
    sig = check_signal(df, funding_rate, ls_ratio)
    price = sig["price"]
    atr = sig["atr"]
    state = read_state()

    funding_text, _ = interpret_funding(funding_rate)
    ls_text = interpret_ls(ls_ratio)
    oi_text = interpret_oi(oi_change)

    fng_text = f"{fng_value}/100 ({fng_label})" if fng_value is not None else "N/A"
    sentiment_block = (
        f"\nSentiment\n"
        f"  Fear & Greed: {fng_text}\n"
        f"  Funding: {funding_text}\n"
        f"  L/S Ratio: {ls_text}\n"
        f"  OI (4H): {oi_text}"
        + (f" ({oi_val}B)" if oi_val else "")
    )

    print(f"Price:{price:.1f} ATR:{atr:.1f}({sig['atr_pct']}%) RSI:{sig['rsi']} Vol:{sig['vol_ratio']}")
    print(f"Score:{sig['score']}/{sig['score_max']} Threshold:{sig['threshold']} Strength:{sig['strength']}")
    print(f"Details: { {k: f'{v[0]}/{v[1]}' for k, v in sig['scores'].items()} }")
    print(f"Position: {json.dumps(state) if state else 'No position'}")

    # ══ No position: check entry ════════════
    if state is None:
        if sig["entry_signal"]:
            stop, target, stop_pct, target_pct = calc_dynamic_levels(price, atr)
            rr = round(abs(target_pct / stop_pct), 1)
            ls_warn = "\nL/S ratio high, watch retail sentiment risk" if sig["ls_warning"] else ""

            score_lines = "\n".join([
                f"  {'[Y]' if v[2] else '[N]'} {k}: {v[0]}/{v[1]}"
                for k, v in sig["scores"].items()
            ])

            new_state = {
                "entry_price": price,
                "initial_stop": stop,
                "take_profit": target,
                "atr": atr,
                "highest": price,
                "trailing_stop": None,
                "trailing_active": False
            }
            write_state(new_state)

            msg = (
                f"ENTRY SIGNAL\n"
                f"{'=' * 22}\n"
                f"{now_str} (Beijing {now_cst_h:02d}:xx)\n\n"
                f"Price: ${price:,.1f}\n\n"
                f"Score: {sig['score']}/{sig['score_max']} - {sig['strength']}\n"
                f"{score_lines}\n\n"
                f"ATR Stop (ATR={atr:.1f} / {sig['atr_pct']}%)\n"
                f"  Stop-loss: ${stop:,} ({stop_pct:.2f}%)\n"
                f"  Take-profit: ${target:,} (+{target_pct:.2f}%)\n"
                f"  Risk/Reward: 1:{rr}\n"
                f"  Trailing stop: activates at +1.5%\n"
                f"{sentiment_block}"
                f"{ls_warn}\n\n"
                f"Suggested position: 30% of account\n"
                f"For reference only"
            )
            send_ntfy(msg, title=f"BTC ENTRY ${price:,.0f}")

        elif sig["tech_signal"] and not sig["funding_ok"]:
            msg = (
                f"SIGNAL FILTERED\n"
                f"{'=' * 22}\n"
                f"{now_str}\n\n"
                f"Technicals: ALL PASS\n"
                f"Price: ${price:,.1f}\n\n"
                f"Blocked by: {sig['funding_block_reason']}\n"
                f"{sentiment_block}\n\n"
                f"Waiting for funding rate to drop below +0.05%"
            )
            send_ntfy(msg, title=f"BTC FILTERED ${price:,.0f}")

        else:
            score_lines = "\n".join([
                f"  {'[Y]' if v[2] else '[N]'} {k}: {v[0]}/{v[1]}"
                for k, v in sig["scores"].items()
            ])
            msg = (
                f"No signal\n"
                f"{'=' * 22}\n"
                f"{now_str}\n\n"
                f"Status: No position, waiting\n"
                f"Price: ${price:,.1f}\n"
                f"ATR: {atr:.1f} ({sig['atr_pct']}%)\n\n"
                f"Score: {sig['score']}/{sig['score_max']} - {sig['strength']}\n"
                f"Entry threshold: {sig['threshold']}\n"
                f"{score_lines}"
                f"{sentiment_block}"
            )
            send_ntfy(msg, title=f"BTC ${price:,.0f} - No signal")

    # ══ Holding: update trailing stop + check exit ═════
    else:
        entry_price = state["entry_price"]
        initial_stop = state["initial_stop"] or entry_price * (1 - STOP_MIN_PCT)
        take_profit = state["take_profit"] or entry_price * (1 + TARGET_MIN_PCT)
        highest = max(state.get("highest", entry_price), price)
        trailing_stop = state.get("trailing_stop")
        trailing_active = state.get("trailing_active", False)
        current_atr = state.get("atr", atr)

        pnl_pct = (price - entry_price) / entry_price * 100

        # Trailing stop logic
        if not trailing_active and pnl_pct >= TRAIL_ACTIVATE_PCT * 100:
            trailing_active = True
            trailing_stop = calc_trailing_stop(highest, current_atr)
            print(f"Trailing stop activated: high {highest:.1f}, stop {trailing_stop:.1f}")
        elif trailing_active:
            new_trail = calc_trailing_stop(highest, current_atr)
            if trailing_stop is None or new_trail > trailing_stop:
                trailing_stop = new_trail

        state["highest"] = highest
        state["trailing_stop"] = trailing_stop
        state["trailing_active"] = trailing_active
        write_state(state)

        # Exit check
        exit_reason = None
        exit_tag = "EXIT"

        if price >= take_profit:
            exit_reason = f"Take-profit hit +{(take_profit/entry_price-1)*100:.1f}%"
            exit_tag = "TARGET HIT"
        elif trailing_active and trailing_stop and price <= trailing_stop:
            locked_pct = (trailing_stop / entry_price - 1) * 100
            exit_reason = f"Trailing stop triggered (locked {locked_pct:+.2f}%)"
            exit_tag = "TRAILING STOP"
        elif price <= initial_stop:
            exit_reason = f"Initial stop hit (ATR x{ATR_STOP_MULT})"
            exit_tag = "STOP LOSS"
        elif sig["trend_broken"]:
            exit_reason = "Price below MA200, trend broken"
            exit_tag = "TREND BREAK"
        elif sig["ma50_broken"] and pnl_pct < 0:
            exit_reason = "Below MA50 with loss, consider reducing"
            exit_tag = "MA50 BREAK"

        if exit_reason:
            msg = (
                f"{exit_tag}\n"
                f"{'=' * 22}\n"
                f"{now_str}\n\n"
                f"Price: ${price:,.1f}\n"
                f"Entry: ${entry_price:,.1f}\n"
                f"High: ${highest:,.1f}\n"
                f"P&L: {pnl_pct:+.2f}%\n\n"
                f"Trigger: {exit_reason}\n"
                f"{sentiment_block}\n\n"
                f"Please decide whether to exit"
            )
            send_ntfy(msg, title=f"BTC {exit_tag} {pnl_pct:+.2f}%")
            write_state(None)

        else:
            trail_line = ""
            if trailing_active and trailing_stop:
                trail_pct = round((trailing_stop / entry_price - 1) * 100, 2)
                dist_trail = round((price / trailing_stop - 1) * 100, 1)
                trail_line = (
                    f"\n  Trailing stop: ${trailing_stop:,} ({trail_pct:+.2f}%, "
                    f"{dist_trail}% from current)\n"
                    f"  Highest: ${highest:,.1f}"
                )
            else:
                activate_gap = round(TRAIL_ACTIVATE_PCT * 100 - pnl_pct, 2)
                trail_line = f"\n  Trailing stop: inactive (need +{activate_gap:.2f}% more)"

            dist_target = round((take_profit / price - 1) * 100, 1)
            dist_stop = round((price / initial_stop - 1) * 100, 1)

            msg = (
                f"POSITION UPDATE\n"
                f"{'=' * 22}\n"
                f"{now_str}\n\n"
                f"Entry: ${entry_price:,.1f}\n"
                f"Current: ${price:,.1f}\n"
                f"P&L: {pnl_pct:+.2f}%\n\n"
                f"Target: ${take_profit:,.1f} (+{dist_target}% away)\n"
                f"Stop: ${initial_stop:,.1f} ({dist_stop}% away)"
                f"{trail_line}"
                f"{sentiment_block}"
            )
            send_ntfy(msg, title=f"BTC Position {pnl_pct:+.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        df = get_klines()
        df = compute_indicators(df)
        funding_rate = get_funding_rate()
        ls_ratio = get_long_short_ratio()
        fng_value, fng_label = get_fear_greed()
        sig = check_signal(df, funding_rate, ls_ratio)
        price = sig["price"]

        score_lines = "\n".join([
            f"  {'[Y]' if v[2] else '[N]'} {k}: {v[0]}/{v[1]}"
            for k, v in sig["scores"].items()
        ])

        state = read_state()
        status = "No position" if state is None else f"Holding (entry ${state['entry_price']:,.1f})"
        fng_text = f"{fng_value}/100 ({fng_label})" if fng_value is not None else "N/A"

        msg = (
            f"Price: ${price:,.1f}\n"
            f"Status: {status}\n"
            f"Fear & Greed: {fng_text}\n\n"
            f"Score: {sig['score']}/{sig['score_max']} (threshold {sig['threshold']})\n"
            f"{score_lines}\n\n"
            f"{'No entry signal' if not sig['entry_signal'] else 'ENTRY SIGNAL ACTIVE!'}"
        )
        send_ntfy(msg, title=f"BTC ${price:,.0f} - {status}")
        print("Test notification sent")
    else:
        main()
