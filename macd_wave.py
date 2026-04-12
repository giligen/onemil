"""
DEPRECATED entry point.

MACD wave now runs inside the unified `onemil-trader` service via main.py.
This file used to own its own process, its own StopMonitor (forced into REST
polling mode due to the 1-data-WS-per-account Alpaca limit), and its own
60s loop. After the unified-streaming refactor (plan: purring-roaming-puppy),
MACD wave is a module inside main.py that shares the bull flag's StopMonitor
and gets WebSocket bar events directly.

To run BOTH strategies:   python main.py --scan --trade
To run only MACD wave:    python main.py --scan --trade --macd
To run only bull flag:    python main.py --scan --trade --flag

The `onemil-macd-wave` systemd service should be stopped and disabled after
the unified `onemil-trader` service has been validated on dev.
"""

import sys


def main():
    sys.stderr.write(
        "\nERROR: macd_wave.py is deprecated.\n\n"
        "MACD wave now runs inside the unified onemil-trader service.\n\n"
        "  Both strategies:   python main.py --scan --trade\n"
        "  MACD wave only:    python main.py --scan --trade --macd\n"
        "  Bull flag only:    python main.py --scan --trade --flag\n\n"
        "The systemd unit `onemil-macd-wave` should be disabled:\n"
        "  sudo systemctl stop onemil-macd-wave\n"
        "  sudo systemctl disable onemil-macd-wave\n"
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
