#!/bin/bash
# Claude Code inside the tmux session "prod-onemil".
#   bash scripts/claude_tmux.sh              attach (creates the session with Claude resumed if missing)
#   bash scripts/claude_tmux.sh --boot       create the session detached, no attach (used by the @reboot cron)
#   CLAUDE_SESSION_ID=<id> bash scripts/...  resume another Claude session
# Detach with Ctrl-b then d; re-attach by running this script again.
set -u
export PATH="/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
TMUX_NAME="prod-onemil"
REPO="/home/ec2-user/onemil"
SESSION_ID="${CLAUDE_SESSION_ID:-257c3e2d-cf38-45d5-94e7-4877f8170f44}"
MODE="${1:-attach}"

if ! tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
    # resume the named session; if that fails, fall back to the most recent one in this directory
    tmux new-session -d -s "$TMUX_NAME" -c "$REPO" \
        "claude --resume $SESSION_ID || claude --continue; exec bash"
    echo "$(date -u +%FT%TZ) created tmux session $TMUX_NAME (claude --resume $SESSION_ID)"
fi
if [ "$MODE" = "--boot" ]; then
    exit 0
fi
exec tmux attach -t "$TMUX_NAME"
