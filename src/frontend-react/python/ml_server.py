#!/usr/bin/env python3
import sys
import json
import traceback
from api_manager import APIManager

manager = None


def send(resp):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def handle_message(msg):
    global manager
    req_id = msg.get("id")
    cmd = msg.get("cmd")
    data = msg.get("data", {})

    try:
        case_id = data.get("case_id")
        if not case_id:
            raise ValueError("Missing case_id")

        # Lazily create the APIManager per process (dummy mode True)
        if manager is None or getattr(manager, "case_id", None) != case_id:
            manager = APIManager(case_id=case_id, dummy=True)

        if cmd == "predict":
            image = data.get("image")
            text = data.get("text_description", "")
            result = manager.get_initial_prediction(image=image, text_description=text, case_id=case_id)
            send({"id": req_id, "ok": True, "result": result})
        elif cmd == "chat":
            question = data.get("question")
            if not question:
                raise ValueError("Missing question")
            result = manager.chat_message(case_id=case_id, user_query=question)
            send({"id": req_id, "ok": True, "result": result})
        else:
            raise ValueError(f"Unknown cmd: {cmd}")
    except Exception as e:
        err = {
            "id": req_id,
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }
        send(err)


def main():
    # Line-delimited JSON protocol
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            send({"id": None, "ok": False, "error": f"Invalid JSON: {e}"})
            continue
        handle_message(msg)


if __name__ == "__main__":
    main()
