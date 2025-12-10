#!/usr/bin/env python3
import sys
import json
import traceback
from datetime import datetime
import cv2
import numpy as np
from api_manager import APIManager

manager = None

# Disclaimer appended to all LLM responses
DISCLAIMER = "\n\n**Please note:** I'm just a helpful assistant and can't give you a medical diagnosis. This information is for general knowledge, and a doctor is the best person to give you a proper diagnosis and treatment plan."


def debug_log(msg: str):
    """Print to stderr so it doesn't interfere with stdout JSON protocol"""
    print(msg, file=sys.stderr, flush=True)


def send(resp):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


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
            has_coin = data.get("has_coin", False)  # Default False if not provided
            debug_log(
                f"[ml_server] add_timeline_entry - received has_coin: {has_coin} (type: {type(has_coin).__name__})"
            )
            debug_log(f"[ml_server] add_timeline_entry - full data keys: {list(data.keys())}")
            if not case_id or not image_path or not date:
                raise ValueError("Missing case_id, image_path, or date")
            APIManager.add_timeline_entry(case_id, image_path, note, date, has_coin)
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
            debug_log(f"[ml_server] predict - received has_coin: {has_coin} (type: {type(has_coin).__name__})")
            debug_log(f"[ml_server] predict - full data keys: {list(data.keys())}")

            if not image_path:
                raise ValueError("Missing image_path")

            # Get predictions and CV analysis before LLM streaming
            # This allows us to:
            #   1) compute a time-tracking CV summary
            #   2) save history
            #   3) send predictionText
            from prediction_texts import get_prediction_text
            from PIL import Image
            import cv2
            import numpy as np

            image = Image.open(image_path)
            saved_image_path = manager._save_image(image)

            # Step 1: Run CV analysis first if needed, then remove coin from image
            cv_analysis = {}
            processed_image = image
            if has_coin:
                debug_log("  → Running CV analysis to detect coin...")
                cv_result = manager._run_cv_analysis(saved_image_path)
                cv_analysis = cv_result

                # Crop to lesion bounding box
                lesion_mask_full = cv_result.get("masks", {}).get("final_mask")
                coin_mask_full = cv_result.get("masks", {}).get("coin_mask_full")
                debug_log(
                    f"  → Lesion mask check: mask is None={lesion_mask_full is None}, has_data={lesion_mask_full is not None and lesion_mask_full.any() if lesion_mask_full is not None else False}"
                )

                if lesion_mask_full is not None and lesion_mask_full.any():
                    debug_log("  → Cropping to lesion bounding box in original image space...")

                    # Get original image dimensions (preserve original quality)
                    original_w, original_h = image.size
                    debug_log(f"  → Original image size: {original_w}x{original_h}")

                    # CV analysis may downscale to max 1024x1024, so we need to scale coordinates back
                    # Get the downscaled mask dimensions
                    mask_h, mask_w = lesion_mask_full.shape[:2]

                    # Calculate scale factor (how much the image was downscaled)
                    # CV analysis downscales based on max dimension to MAX_PROCESS_DIMENSION (1024)
                    max_original_dim = max(original_w, original_h)
                    max_mask_dim = max(mask_w, mask_h)

                    # If the original image was larger than MAX_PROCESS_DIMENSION, it was downscaled
                    if max_original_dim > 1024:
                        # Calculate scale based on how much the max dimension was reduced
                        scale_factor = max_original_dim / max_mask_dim if max_mask_dim > 0 else 1.0
                    else:
                        # Image wasn't downscaled, scale factor is 1.0
                        scale_factor = 1.0

                    debug_log(f"  → Mask size: {mask_w}x{mask_h}")
                    debug_log(f"  → Scale factor (original/mask): {scale_factor:.3f}")

                    # Find bounding box of lesion in mask space
                    lesion_coords = np.where(lesion_mask_full > 0)
                    if len(lesion_coords[0]) > 0:
                        # Get bounding box in mask space
                        lesion_y_min_mask, lesion_y_max_mask = lesion_coords[0].min(), lesion_coords[0].max()
                        lesion_x_min_mask, lesion_x_max_mask = lesion_coords[1].min(), lesion_coords[1].max()

                        # Scale bounding box back to original image space
                        lesion_x_min = int(lesion_x_min_mask * scale_factor)
                        lesion_x_max = int(lesion_x_max_mask * scale_factor)
                        lesion_y_min = int(lesion_y_min_mask * scale_factor)
                        lesion_y_max = int(lesion_y_max_mask * scale_factor)

                        # Add padding around lesion bounding box (in original image space)
                        padding = int(50 * scale_factor)  # Scale padding proportionally
                        crop_x = max(0, lesion_x_min - padding)
                        crop_y = max(0, lesion_y_min - padding)
                        crop_w = min(original_w, lesion_x_max + padding) - crop_x
                        crop_h = min(original_h, lesion_y_max + padding) - crop_y

                        debug_log(
                            f"  → Lesion bounding box (mask space): x=[{lesion_x_min_mask}, {lesion_x_max_mask}], y=[{lesion_y_min_mask}, {lesion_y_max_mask}]"
                        )
                        debug_log(
                            f"  → Lesion bounding box (original space): x=[{lesion_x_min}, {lesion_x_max}], y=[{lesion_y_min}, {lesion_y_max}]"
                        )
                        debug_log(f"  → Crop region (original space): x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

                        # Ensure we have a valid crop (at least some pixels)
                        if crop_w > 0 and crop_h > 0:
                            # Crop directly from original PIL image (preserves quality and color space)
                            processed_image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                            debug_log(
                                f"  → ✅ Cropped image to lesion bounding box: {crop_w}x{crop_h} (original quality preserved)"
                            )

                            # Save overlay and processed image for debugging (convert to OpenCV only for visualization)
                            try:
                                # Convert to numpy for overlay visualization
                                img_array = np.array(image.convert("RGB"))
                                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                                # Resize masks to original image size for overlay
                                lesion_mask_original = cv2.resize(
                                    lesion_mask_full, (original_w, original_h), interpolation=cv2.INTER_NEAREST
                                )

                                # Create overlay
                                overlay_img = img_bgr.copy()
                                lesion_overlay = np.zeros_like(overlay_img)
                                lesion_overlay[lesion_mask_original > 0] = [0, 255, 0]  # Green in BGR
                                overlay_img = cv2.addWeighted(overlay_img, 0.7, lesion_overlay, 0.3, 0)

                                if coin_mask_full is not None:
                                    coin_mask_original = cv2.resize(
                                        coin_mask_full, (original_w, original_h), interpolation=cv2.INTER_NEAREST
                                    )
                                    coin_overlay = np.zeros_like(overlay_img)
                                    coin_overlay[coin_mask_original > 0] = [0, 0, 255]  # Red in BGR
                                    overlay_img = cv2.addWeighted(overlay_img, 0.7, coin_overlay, 0.3, 0)

                                cv2.imwrite("/Users/tk20/Downloads/coin_mask_overlay.png", overlay_img)
                                debug_log("  → ✅ Saved mask overlay to /Users/tk20/Downloads/coin_mask_overlay.png")

                                # Save cropped image (from PIL, preserving quality)
                                processed_image.save("/Users/tk20/Downloads/processed_image.png", "PNG")
                                debug_log("  → ✅ Saved cropped image to /Users/tk20/Downloads/processed_image.png")
                            except Exception as e:
                                debug_log(f"  → ❌ Failed to save overlay/processed image: {e}")
                        else:
                            debug_log("  → ⚠️ Invalid crop dimensions, skipping crop")
                    else:
                        debug_log("  → ⚠️ Could not find lesion coordinates in mask")
                else:
                    debug_log("  → ⚠️ No coin detected or coin mask is empty, skipping coin removal")

            # Step 2: Run local ML model for embeddings (using coin-removed image)
            embedding = manager._run_local_ml_model(processed_image)
            updated_text_description = manager.update_text_input(text)
            predictions_raw = manager._run_cloud_ml_model(embedding, updated_text_description)

            # Check if predictions_raw is a dict (new format) or list (old format)
            if isinstance(predictions_raw, dict):
                # New format: already a dict
                predictions = predictions_raw
            else:
                # Old format: list of {"class": ..., "probability": ...}
                predictions = {item["class"]: item["probability"] for item in predictions_raw}

            # Check if model returned UNCERTAIN
            if "UNCERTAIN" in predictions:
                debug_log("⚠️ Model is uncertain about this prediction")
                # Log the top predictions even when uncertain (excluding UNCERTAIN itself)
                other_predictions = {k: v for k, v in predictions.items() if k != "UNCERTAIN"}
                if other_predictions:
                    debug_log("Top predictions (excluding UNCERTAIN):")
                    for disease, prob in sorted(other_predictions.items(), key=lambda x: x[1], reverse=True):
                        debug_log(f"  - {disease}: {prob:.4f}")
                send(
                    {
                        "id": req_id,
                        "ok": False,
                        "error": "UNCERTAIN",
                        "message": "Unable to determine the condition from this image. Please try again with better lighting, a clearer photo, or more descriptive text. It's also possible that there's nothing concerning to identify.",
                    }
                )
                return

            # Get top prediction label for later use
            top_prediction_label = max(predictions.items(), key=lambda x: x[1])[0] if predictions else ""

            # Step 3: Generate time-tracking summary from CV + history
            tracking_summary = ""
            if has_coin and cv_analysis:
                tracking_summary = manager._get_time_tracking_summary(
                    predictions=predictions,
                    text_description=text,  # ORIGINAL user input
                    cv_analysis=cv_analysis,
                )

            # Save history entry (including tracking summary) BEFORE we start LLM streaming,
            # so that the TimeTrackingPanel can read it as soon as the UI switches to results.
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Clean cv_analysis to remove numpy arrays for JSON serialization
            cv_analysis_clean = {}
            if cv_analysis:
                for key, value in cv_analysis.items():
                    if key == "masks" or key == "images":
                        # Skip numpy arrays (masks and images)
                        continue
                    elif key == "coin_data" and value is not None:
                        # Convert tuple to list for JSON serialization
                        cv_analysis_clean[key] = list(value) if isinstance(value, tuple) else value
                    else:
                        cv_analysis_clean[key] = value

            # Convert numpy types (int64, float64, etc.) to Python native types
            cv_analysis_clean = _convert_numpy_types(cv_analysis_clean)

            manager._save_history_entry(
                date=current_date,
                cv_analysis=cv_analysis_clean,
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
            # Append disclaimer to the response
            llm_response = llm_response + DISCLAIMER

            # Save conversation (history is already saved above)
            # Save ORIGINAL user text (without demographics) to conversation
            manager._save_conversation_entry(
                user_message=text,  # Use original text, not augmented
                llm_response=llm_response,
                user_timestamp=user_timestamp,
                llm_timestamp=llm_timestamp,
            )

            # Build enriched disease and return complete result
            enriched_disease = manager._build_enriched_disease()

            # Clean cv_analysis to remove numpy arrays for JSON serialization
            cv_analysis_clean = {}
            if cv_analysis:
                for key, value in cv_analysis.items():
                    if key == "masks" or key == "images":
                        # Skip numpy arrays (masks and images)
                        continue
                    elif key == "coin_data" and value is not None:
                        # Convert tuple to list for JSON serialization
                        cv_analysis_clean[key] = list(value) if isinstance(value, tuple) else value
                    else:
                        cv_analysis_clean[key] = value

            # Convert numpy types (int64, float64, etc.) to Python native types
            cv_analysis_clean = _convert_numpy_types(cv_analysis_clean)

            result = {
                "llm_response": llm_response,
                "predictions": predictions,
                "cv_analysis": cv_analysis_clean,
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
