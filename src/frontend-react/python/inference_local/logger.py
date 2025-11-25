import sys

class SimpleLogger:
    @staticmethod
    def info(msg):
        print(f"[INFO] {msg}", file=sys.stderr, flush=True)
    
    @staticmethod
    def warning(msg):
        print(f"[WARNING] {msg}", file=sys.stderr, flush=True)
    
    @staticmethod
    def error(msg):
        print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

logger = SimpleLogger()
