#!/bin/bash
#
# Nightly Bull Flag BT cache refresh.
#
# Extends the production raw-trade cache (bull_flag_cache_e50_x30.csv)
# forward by appending newly-built rows for days after the last cached date.
#
# Approach (chosen after auditing batch_backtest.py paths):
#   1. Find LAST_CACHED date in production cache.
#   2. If TODAY > LAST_CACHED, run --build-cache for the gap into a temp
#      file (this disables max_trades/regime filters → produces raw
#      Stage-1 data, matching the existing cache's nature).
#   3. Append only the data rows (skip header) from the temp file to the
#      production cache. Atomic via tmp+mv.
#   4. If 0 trades were generated for the missing range (quiet market),
#      no-op the append. The cache stays at LAST_CACHED — next run will
#      retry. To prevent infinite re-attempts of empty days, we also
#      track date-coverage in a sentinel file.
#
# Why NOT use the auto-build path (no --build-cache flag):
#   - Auto-build at batch_backtest.py:2521 runs with production filters
#     (max_trades_per_day=5, regime_sizing on, daily_loss_limit) which
#     produce a Stage-2 filtered cache. The production cache is Stage-1
#     raw — mixing filtered + raw rows would break the schema invariant.
#
# Why NOT --build-cache against the production cache directly:
#   - --build-cache opens the file with mode 'w' (line 2795). It would
#     OVERWRITE the existing 2825 rows of historical data. Disastrous.
#
# Logs to: logs/nightly_bt_update.log
#
set -uo pipefail

REPO=/home/ec2-user/onemil
LOG="${REPO}/logs/nightly_bt_update.log"
CACHE_PATH="${REPO}/data/bull_flag_cache_e50_x30.csv"
# Coverage sentinel (2026-08-28): written on every SUCCESSFUL run —
# including zero-row quiet days — so bf_decision_parity can distinguish
# "built, produced 0 rows" (BT side KNOWN-zero) from "job failed/stale"
# (BT side unknowable). 1/3 of trading days produce zero raw rows; before
# this, every such day was thrown away as BT_STALE.
COVERAGE_SENTINEL="${REPO}/data/bull_flag_cache_coverage.txt"
TMP_CACHE="/tmp/nightly_bt_$(date -u +%s).csv"

cd "$REPO" || exit 1

# Stale cleanup — if a previous run died mid-rename, .tmp may be left over.
rm -f "${CACHE_PATH}.tmp"

mark_log() {
    {
        echo ""
        echo "================================================================"
        echo "Nightly BT update: $1 — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "================================================================"
    } >> "$LOG"
}

# Telegram alert on failure (2026-05-14). This cron runs outside the
# onemil-trader service so its log has no Telegram relay — without this, a
# wedged nightly cron (e.g. the 2026-05-08 schema-mismatch failure) sits
# unnoticed in the log file for weeks. Non-fatal: `|| true` so a failed
# alert never masks the real exit code.
alert_failure() {
    /usr/bin/python3 "${REPO}/scripts/send_telegram_alert.py" \
        "🔴 nightly_bt_update FAILED: $1 — $(date -u '+%Y-%m-%d %H:%M UTC'). Cache left untouched; manual rebuild needed (batch_backtest.py --build-cache)." \
        >> "$LOG" 2>&1 || true
}

mark_log "started"

if [ ! -f "$CACHE_PATH" ]; then
    echo "ERROR: cache not found at $CACHE_PATH — refusing to bootstrap." >> "$LOG"
    exit 2
fi

TODAY=$(date -u +%Y-%m-%d)
LAST_CACHED=$(awk -F, 'NR>1 {print $2}' "$CACHE_PATH" | sort -u | tail -1)

echo "TODAY=$TODAY  LAST_CACHED=$LAST_CACHED" >> "$LOG"

# Source-of-truth for "do we need to build?" is the cache itself, not a
# sentinel. If the cache already covers TODAY, exit. Self-corrects: if a
# pre-market run produced no rows for TODAY, the next run still attempts.
if [[ ! "$TODAY" > "$LAST_CACHED" ]]; then
    echo "TODAY ($TODAY) is not after LAST_CACHED ($LAST_CACHED) — nothing to build." >> "$LOG"
    echo "$TODAY" > "$COVERAGE_SENTINEL"
    mark_log "no-op (nothing to build)"
    exit 0
fi

# Compute next-day-after-LAST_CACHED as the build start
NEW_START=$(date -u -d "$LAST_CACHED + 1 day" +%Y-%m-%d)
echo "Building gap: $NEW_START to $TODAY → $TMP_CACHE" >> "$LOG"

# --build-cache disables max_trades/regime/CB filters → produces raw cache
# matching the schema of the production cache.
#
# Sizing args MUST mirror live config.yaml so cached pnl values are
# directly comparable to live trades. Mismatch produced apples-to-oranges
# BT-vs-LIVE numbers for week of 2026-05-04 (cache had ~$240/trade risk;
# live ran $60/trade × tier multipliers). config.yaml as of 2026-05-08:
#   trading.capital: 5000
#   trading.risk_per_trade: 60
#   trading.max_shares: 15000
/usr/bin/python3 batch_backtest.py \
    --start "$NEW_START" --end "$TODAY" \
    --build-cache --no-cache \
    --cache-file "$TMP_CACHE" \
    --capital 5000 --risk 60 --max-shares 15000 \
    >> "$LOG" 2>&1

BUILD_EXIT=$?
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "ERROR: build exited $BUILD_EXIT — leaving cache untouched." >> "$LOG"
    rm -f "$TMP_CACHE"
    mark_log "FAILED (build exit=$BUILD_EXIT)"
    alert_failure "build step exited $BUILD_EXIT"
    exit "$BUILD_EXIT"
fi

if [ ! -f "$TMP_CACHE" ]; then
    echo "WARNING: temp cache not produced. No data to append." >> "$LOG"
    mark_log "no-op (no temp cache produced)"
    exit 0
fi

NEW_ROWS=$(($(wc -l < "$TMP_CACHE") - 1))
echo "Temp cache has $NEW_ROWS new trade rows." >> "$LOG"

if [ "$NEW_ROWS" -gt 0 ]; then
    # SCHEMA GUARD (added 2026-05-01 after CSV_HEADERS-grew incident)
    # ===============================================================
    # If the temp build was produced with a different CSV_HEADERS than the
    # production cache (typical cause: schema grew between when production
    # was last fully rebuilt and now), naive `tail >> file` would silently
    # produce a file with mismatched header vs body row widths. Downstream
    # csv.DictReader then drops fields, BT loses data, alignment work goes
    # off the rails (see commit history 2026-05-01).
    #
    # Refuse to append if column counts disagree. The operator must either
    # rebuild the production cache, or run scripts/migrate_cache_schema.py
    # (which we should add if this fires more than once).
    PROD_COLS=$(head -1 "$CACHE_PATH" | awk -F, '{print NF}')
    TMP_COLS=$(head -1 "$TMP_CACHE" | awk -F, '{print NF}')
    if [ "$PROD_COLS" != "$TMP_COLS" ]; then
        {
            echo ""
            echo "ERROR: schema mismatch — production cache header has $PROD_COLS columns,"
            echo "       temp build has $TMP_COLS columns. Refusing to append (would corrupt"
            echo "       the cache). Production cache LEFT UNTOUCHED."
            echo ""
            echo "       Production header (first 5): $(head -1 "$CACHE_PATH" | cut -d, -f1-5)"
            echo "       Temp build header (first 5): $(head -1 "$TMP_CACHE" | cut -d, -f1-5)"
            echo ""
            echo "       To recover: either rebuild the production cache via --build-cache"
            echo "       directly (overwrites — last-resort), or migrate the prod cache"
            echo "       header to current CSV_HEADERS and pad short rows with empty fields."
        } >> "$LOG"
        rm -f "$TMP_CACHE"
        mark_log "FAILED (schema mismatch: prod=$PROD_COLS, tmp=$TMP_COLS)"
        alert_failure "cache schema mismatch (prod=$PROD_COLS cols, build=$TMP_COLS cols)"
        exit 3
    fi

    # Atomic append: copy production cache to .tmp, append new rows, rename.
    cp "$CACHE_PATH" "${CACHE_PATH}.tmp"
    tail -n +2 "$TMP_CACHE" >> "${CACHE_PATH}.tmp"
    mv "${CACHE_PATH}.tmp" "$CACHE_PATH"
    echo "Appended $NEW_ROWS rows to $CACHE_PATH (schema OK at $PROD_COLS cols)" >> "$LOG"
fi

rm -f "$TMP_CACHE"

NEW_LAST=$(awk -F, 'NR>1 {print $2}' "$CACHE_PATH" | sort -u | tail -1)
NEW_TOTAL=$(($(wc -l < "$CACHE_PATH") - 1))
echo "Cache now has $NEW_TOTAL rows; last cached date: $NEW_LAST" >> "$LOG"

# Success (incl. zero-row builds): the gap NEW_START..TODAY was built.
echo "$TODAY" > "$COVERAGE_SENTINEL"
mark_log "completed (added $NEW_ROWS rows)"
exit 0
