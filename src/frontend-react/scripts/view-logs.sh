#!/bin/bash
# Helper script to view app logs

LOG_FILE="/tmp/pibu_ai_debug.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   The app may not be running or hasn't written logs yet."
    exit 1
fi

echo "📋 App Logs: $LOG_FILE"
echo "=========================================="
echo ""

case "${1:-tail}" in
    tail)
        echo "Showing last 50 lines (use 'tail -f' for real-time):"
        echo ""
        tail -50 "$LOG_FILE"
        ;;
    follow|f)
        echo "Following log file (Ctrl+C to stop)..."
        echo ""
        tail -f "$LOG_FILE"
        ;;
    all|a)
        echo "Showing entire log file:"
        echo ""
        cat "$LOG_FILE"
        ;;
    search|s)
        if [ -z "$2" ]; then
            echo "Usage: $0 search <pattern>"
            exit 1
        fi
        echo "Searching for: $2"
        echo ""
        grep -i "$2" "$LOG_FILE" | tail -50
        ;;
    errors|e)
        echo "Showing errors only:"
        echo ""
        grep -i "error\|exception\|failed\|fail" "$LOG_FILE" | tail -50
        ;;
    python|p)
        echo "Showing Python process logs:"
        echo ""
        grep -i "python\|cv_analysis\|api_manager\|ml_server" "$LOG_FILE" | tail -50
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)     - Show last 50 lines (default)"
        echo "  follow, f     - Follow log file in real-time"
        echo "  all, a        - Show entire log file"
        echo "  search, s     - Search for pattern (requires pattern)"
        echo "  errors, e     - Show errors only"
        echo "  python, p     - Show Python-related logs"
        echo ""
        echo "Examples:"
        echo "  $0              # Last 50 lines"
        echo "  $0 follow       # Real-time logs"
        echo "  $0 search error # Search for 'error'"
        echo "  $0 errors       # Show only errors"
        ;;
esac

