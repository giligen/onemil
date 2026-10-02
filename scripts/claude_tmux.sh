#!/bin/bash
# Attach (or start) the Claude Code session inside the tmux session "prod-onemil".
# Usage:  bash scripts/claude_tmux.sh            # attach, creating it if needed
#         bash scripts/claude_tmux.sh <session>  # resume a specific Claude session id
# Detach with Ctrl-b then d; re-attach by running this script again.
set -u
TMUX_NAME="prod-onemil"
REPO="/home/ec2-user/onemil"
SESSION_ID="${1:-257c3e2d-cf38-45d5-94e7-4877f8170f44}"

if tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
    exec tmux attach -t "$TMUX_NAME"
fi
tmux new-session -d -s "$TMUX_NAME" -c "$REPO" "claude --resume $SESSION_ID; exec bash"
exec tmux attach -t "$TMUX_NAME"
