#!/usr/bin/env python3
import sys
import json
import traceback
from datetime import datetime
from api_manager import APIManager

manager = None


def debug_log(msg: str):
    """Print to stderr so it doesn't interfere with stdout JSON protocol"""
    print(msg, file=sys.stderr, flush=True)


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
        elif cmd == "delete_cases":
            case_ids = data.get("case_ids", [])
            if not case_ids:
                raise ValueError("Missing case_ids")
            APIManager.delete_cases(case_ids)
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
            has_coin = data.get("has_coin", False)  # Default False if not provided

            if not image_path:
                raise ValueError("Missing image_path")

            # Get predictions and CV analysis before LLM streaming
            # This allows us to:
            #   1) compute a time-tracking CV summary
            #   2) save history
            #   3) send predictionText
            from prediction_texts import get_prediction_text
            from PIL import Image

            image = Image.open(image_path)
            saved_image_path = manager._save_image(image)
            embedding = manager._run_local_ml_model(image)
            updated_text_description = manager.update_text_input(text)
            predictions_raw = manager._run_cloud_ml_model(embedding, updated_text_description)
            predictions = {item["class"]: item["probability"] for item in predictions_raw}

            # Determine if we should run CV analysis
            top_prediction_label = max(predictions.items(), key=lambda x: x[1])[0] if predictions else ""
            should_run_cv = has_coin

            # Run CV analysis if conditions are met
            cv_analysis = {}
            tracking_summary = ""
            if should_run_cv:
                cv_analysis = manager._run_cv_analysis(saved_image_path)
                # Generate time-tracking summary text from CV + history
                tracking_summary = manager._get_time_tracking_summary(
                    predictions=predictions,
                    text_description=text,  # ORIGINAL user input
                    cv_analysis=cv_analysis,
                )
            else:
                debug_log(f"[ml_server] Skipping CV analysis (has_coin={has_coin}, top_prediction={top_prediction_label})")

            # Save history entry (including tracking summary) BEFORE we start LLM streaming,
            # so that the TimeTrackingPanel can read it as soon as the UI switches to results.
            current_date = datetime.now().strftime("%Y-%m-%d")
            manager._save_history_entry(
                date=current_date,
                cv_analysis=cv_analysis,
                predictions=predictions,
                image_path=saved_image_path,
                text_summary=text,  # ORIGINAL user input
                tracking_summary=tracking_summary,
            )

            # Get predictionText immediately after predictions & tracking summary are ready
            predictionText = ""
            if predictions and top_prediction_label:
                predictionText = get_prediction_text(top_prediction_label)
                debug_log(f"[ml_server] Generated predictionText for {top_prediction_label}: {predictionText[:100]}...")

            # Send predictionText metadata before streaming starts
            if predictionText:
                debug_log(f"[ml_server] Sending predictionText to frontend (req_id={req_id})")
                send({"id": req_id, "predictionText": predictionText})
                debug_log("[ml_server] predictionText sent successfully")

            # Create streaming callback for explanation chunks
            def on_stream_chunk(text: str):
                # Each call becomes one "chunk" event up to Node
                send({"id": req_id, "chunk": text})

            # Continue with LLM call (streaming if on_chunk provided).
            # Pass high-level CV tracking summary text into metadata instead of raw CV metrics.
            metadata = {
                "user_input": updated_text_description,
                "cv_tracking_summary": tracking_summary,
                "history": manager.case_history["dates"],
            }
            llm_response_dict, llm_timestamp = manager._call_llm_explain(
                predictions=predictions,
                metadata=metadata,
                on_chunk=on_stream_chunk,
            )
            llm_response = (
                llm_response_dict.get("answer", "") if isinstance(llm_response_dict, dict) else str(llm_response_dict)
            )

            # Save conversation (history is already saved above)
            manager._save_conversation_entry(
                user_message=updated_text_description,
                llm_response=llm_response,
                user_timestamp=user_timestamp,
                llm_timestamp=llm_timestamp,
            )

            # Build enriched disease and return complete result
            enriched_disease = manager._build_enriched_disease()
            result = {
                "llm_response": llm_response,
                "predictions": predictions,
                "cv_analysis": cv_analysis,
                "embedding": embedding,
                "text_description": text,
                "enriched_disease": enriched_disease,
            }

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
        elif cmd == "load_conversation_history":
            # Return the conversation history for this case
            conversation_history = manager.conversation_history or []
            send({"id": req_id, "ok": True, "result": conversation_history})

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
