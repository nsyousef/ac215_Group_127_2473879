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
        # Static commands that don't need case_id or manager instance
        if cmd == "save_demographics":
            demographics_data = data.get("demographics")
            if not demographics_data:
                raise ValueError("Missing demographics data")
            APIManager.save_demographics(demographics_data)
            send({"id": req_id, "ok": True, "result": {"success": True}})
            return
        elif cmd == "load_demographics":
            result = APIManager.load_demographics()
            send({"id": req_id, "ok": True, "result": result})
            return
        elif cmd == "save_body_location":
            case_id = data.get("case_id")
            body_location = data.get("body_location")
            if not case_id or not body_location:
                raise ValueError("Missing case_id or body_location")
            APIManager.save_body_location(case_id, body_location)
            send({"id": req_id, "ok": True, "result": {"success": True}})
            return
        elif cmd == "reset_all_data":
            APIManager.reset_all_data()
            send({"id": req_id, "ok": True, "result": {"success": True}})
            return
        elif cmd == "load_diseases":
            diseases = APIManager.load_diseases()
            send({"id": req_id, "ok": True, "result": {"diseases": diseases}})
            return
        elif cmd == "save_diseases":
            diseases = data.get("diseases", [])
            APIManager.save_diseases(diseases)
            send({"id": req_id, "ok": True, "result": {"success": True}})
            return
        elif cmd == "load_case_history":
            case_id = data.get("case_id")
            if not case_id:
                raise ValueError("Missing case_id")
            history = APIManager.load_case_history(case_id)
            send({"id": req_id, "ok": True, "result": history})
            return
        elif cmd == "save_case_history":
            case_id = data.get("case_id")
            case_history = data.get("case_history")
            if not case_id or not case_history:
                raise ValueError("Missing case_id or case_history")
            APIManager.save_case_history(case_id, case_history)
            send({"id": req_id, "ok": True, "result": {"success": True}})
            return
        elif cmd == "update_disease_name":
            case_id = data.get("case_id")
            name = data.get("name")
            if not case_id or not name:
                raise ValueError("Missing case_id or name")
            APIManager.update_disease_name(case_id, name)
            send({"id": req_id, "ok": True, "result": {"success": True}})
            return
        elif cmd == "add_timeline_entry":
            case_id = data.get("case_id")
            image_path = data.get("image_path")
            note = data.get("note", "")
            date = data.get("date")
            if not case_id or not image_path or not date:
                raise ValueError("Missing case_id, image_path, or date")
            APIManager.add_timeline_entry(case_id, image_path, note, date)
            send({"id": req_id, "ok": True, "result": {"success": True}})
            return

        # Commands that need case_id and manager instance
        case_id = data.get("case_id")
        if not case_id:
            raise ValueError("Missing case_id")

        # Lazily create the APIManager per process (dummy mode False)
        if manager is None or getattr(manager, "case_id", None) != case_id:
            manager = APIManager(case_id=case_id, dummy=False)

        if cmd == "predict":
            image_path = data.get("image_path")
            text = data.get("text_description", "")
            user_timestamp = data.get("user_timestamp")

            if not image_path:
                raise ValueError("Missing image_path")

            # Create streaming callback for explanation chunks
            def on_stream_chunk(text: str):
                # Each call becomes one "chunk" event up to Node
                send({"id": req_id, "chunk": text})

            # Run prediction (this will stream explanation chunks)
            result = manager.get_initial_prediction(
                image_path=image_path,
                text_description=text,
                user_timestamp=user_timestamp,
                on_chunk=on_stream_chunk,  # ← IMPORTANT: Pass streaming callback
            )

            # After streaming, send final combined response
            send({"id": req_id, "ok": True, "result": result})
        elif cmd == "chat":
            question = data.get("question")
            user_timestamp = data.get("user_timestamp")
            if not question:
                raise ValueError("Missing question")

            # This is what actually turns on streaming for this request
            def on_stream_chunk(text: str):
                # Each call becomes one "chunk" event up to Node
                send({"id": req_id, "chunk": text})

            result = manager.chat_message(
                user_query=question,
                user_timestamp=user_timestamp,
                on_chunk=on_stream_chunk,  # ← IMPORTANT
            )

            # Final message with the full structured result
            send({"id": req_id, "ok": True, "result": result})

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
