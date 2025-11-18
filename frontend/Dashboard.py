# Dashboard.py
import json
import time
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger
import pandas as pd

# Update to your FastAPI endpoint if different
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

LOGGER = get_logger(__name__)

# Default feature list for the wine dataset (13 features)
WINE_FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280_od315_of_diluted_wines", "proline"
]

# Map numeric class index to human friendly labels (edit to suit)
CLASS_INDEX_TO_LABEL = {
    0: "Cultivar A (class_0)",
    1: "Cultivar B (class_1)",
    2: "Cultivar C (class_2)"
}

def call_api(endpoint: str, payload: dict, timeout: float = 15.0):
    """
    Call the FastAPI endpoint and return (response object, elapsed_seconds) or raise.
    """
    start = time.time()
    resp = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}{endpoint}", json=payload, timeout=timeout)
    elapsed = time.time() - start
    return resp, elapsed

def try_batch_with_fallback(endpoint: str, records):
    """
    Try sending `records` (a list) as a single batch to `endpoint`.
    If the server rejects with the 'Input should be a valid dictionary or object' error,
    fallback to calling endpoint once per record and aggregate responses.
    Returns (success, result, elapsed_total_seconds)
      - success: True if batch or per-item calls succeeded
      - result: JSON response (list) or error message
      - elapsed_total_seconds: sum of elapsed times
    """
    elapsed_total = 0.0

    # 1) Try sending as single payload (list at top-level)
    try:
        resp, elapsed = call_api(endpoint, records)
        elapsed_total += elapsed
    except Exception as e:
        return False, f"Request failed: {e}", elapsed_total

    # If batch accepted
    if isinstance(resp, requests.Response) and resp.status_code == 200:
        try:
            return True, resp.json(), elapsed_total
        except Exception:
            return True, {"raw_text": resp.text}, elapsed_total

    # If server returned 422 with the model_attributes_type message -> fallback
    if isinstance(resp, requests.Response) and resp.status_code == 422:
        # try to parse error detail safely
        try:
            err = resp.json()
            # check message text to detect this specific server expectation
            detail = err.get("detail", [])
            should_fallback = False
            for d in detail:
                if isinstance(d, dict) and "msg" in d and "dictionary or object" in str(d["msg"]):
                    should_fallback = True
                    break
        except Exception:
            should_fallback = False

        if should_fallback:
            # Call endpoint once per record and collect results
            aggregated = []
            for rec in records:
                try:
                    single_resp, single_elapsed = call_api(endpoint, rec)
                    elapsed_total += single_elapsed
                except Exception as e:
                    aggregated.append({"error": f"Request failed: {e}", "input": rec})
                    continue

                if isinstance(single_resp, requests.Response) and single_resp.status_code == 200:
                    try:
                        aggregated.append(single_resp.json())
                    except:
                        aggregated.append({"raw_text": single_resp.text})
                else:
                    # collect error info
                    try:
                        aggregated.append({"error": single_resp.json(), "status": single_resp.status_code})
                    except:
                        aggregated.append({"error": single_resp.text, "status": getattr(single_resp, "status_code", None)})
            return True, aggregated, elapsed_total

    # else return the original error
    try:
        return False, resp.json(), elapsed_total
    except Exception:
        return False, resp.text if isinstance(resp, requests.Response) else str(resp), elapsed_total


def normalize_uploaded_json(uploaded_json):
    """
    Normalize uploaded JSON into either:
      - a single flat record dict of features, or
      - a list of flat record dicts (for batch)
    Accepts:
      {"input": {...}}  OR {...flat...}  OR {"input_batch":[{...},{...}]} OR [...]
    """
    if isinstance(uploaded_json, dict):
        if "input_batch" in uploaded_json and isinstance(uploaded_json["input_batch"], list):
            return uploaded_json["input_batch"]
        if "input" in uploaded_json and isinstance(uploaded_json["input"], dict):
            return uploaded_json["input"]
        # else assume dict is the flat feature dict
        return uploaded_json
    elif isinstance(uploaded_json, list):
        return uploaded_json
    else:
        raise ValueError("Uploaded JSON not recognized. Provide a dict or a list of dicts.")

def run():
    st.set_page_config(page_title="Wine Classification Dashboard", layout="wide")
    st.title("Wine Classification Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Backend status")
        try:
            r = requests.get(FASTAPI_BACKEND_ENDPOINT)
            if r.status_code == 200:
                st.success("Backend online ✅")
            else:
                st.warning(f"Backend responded: {r.status_code}")
        except Exception as e:
            LOGGER.error("Backend check failed", e)
            st.error("Backend offline ❌")

        st.markdown("---")
        st.header("Input options")
        input_mode = st.radio("Choose input mode", ["Manual (single)", "JSON upload (single)", "JSON upload (batch)"])

        # Manual input uses wine features by default
        manual_inputs = {}
        if input_mode == "Manual (single)":
            st.info("Enter feature values (wine features)")
            cols = st.columns(2)
            for i, f in enumerate(WINE_FEATURES):
                with cols[i % 2]:
                    # choose reasonable defaults; user can change
                    if f in ("magnesium", "proline"):
                        manual_inputs[f] = st.number_input(f, min_value=0.0, value=100.0, step=1.0)
                    else:
                        manual_inputs[f] = st.number_input(f, min_value=0.0, value=1.0, step=0.01)
        else:
            uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
            uploaded_json = None
            if uploaded_file:
                try:
                    uploaded_json = json.load(uploaded_file)
                    st.success("JSON loaded")
                    st.json(uploaded_json)
                    # save to session for main scope use
                    st.session_state["UPLOADED_JSON"] = uploaded_json
                except Exception as e:
                    st.error("Failed to parse JSON: " + str(e))
                    LOGGER.error(e)
                    st.session_state["UPLOADED_JSON"] = None
            else:
                st.session_state["UPLOADED_JSON"] = None

        st.markdown("---")
        st.write("Choose endpoint to call:")
        endpoint_choice = st.selectbox("Endpoint", ["/predict (single)", "/predict-with-probability"])
        st.write("Note: use `/predict-with-probability` for probabilities and batch responses.")

        st.markdown("---")
        st.button("Clear results", key="clear_results")

    # Main body
    st.header("Make a Prediction")
    st.write("Use the sidebar to set inputs, upload JSON, and choose endpoint.")

    # Action buttons
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        predict_btn = st.button("Predict (single)")
    with col2:
        proba_btn = st.button("Predict with probabilities")
    with col3:
        batch_btn = st.button("Batch predict (upload list)")

    # Prepare payload for single prediction
    payload = None
    # Priority: if manual -> use manual inputs. Else use uploaded JSON from session.
    if input_mode == "Manual (single)":
        payload = manual_inputs
    else:
        uploaded_json = st.session_state.get("UPLOADED_JSON")
        if uploaded_json:
            try:
                normalized = normalize_uploaded_json(uploaded_json)
                payload = normalized
            except Exception as e:
                st.error(str(e))
                payload = None
        else:
            payload = None

    # Display input preview
    st.subheader("Input preview")
    if payload is None:
        st.info("No input ready. Enter manual inputs or upload a JSON file in the sidebar.")
    else:
        # show single or first record for preview
        if isinstance(payload, list):
            st.json({"input_preview": payload[0]})
        else:
            st.json({"input_preview": payload})

    # Results container
    result_container = st.empty()

    # Helper to process response and display
    def process_response(resp_obj, elapsed_time):
        try:
            if resp_obj.status_code != 200:
                # attempt to decode error detail
                try:
                    err = resp_obj.json()
                except:
                    err = resp_obj.text
                result_container.error(f"Server returned {resp_obj.status_code}: {err}")
                return

            out = resp_obj.json()
            # Show raw output
            st.subheader("Raw API response")
            st.json(out)
            # show inference time
            st.write(f"⏱️ Inference time: **{elapsed_time:.3f} s**")

            # prediction index and class_name (if available)
            pred_index = out.get("prediction")
            api_class_name = out.get("class_name")
            st.markdown("### Prediction")
            st.write(f"**Index:** {pred_index}")
            st.write(f"**API class_name:** {api_class_name}")

            friendly = CLASS_INDEX_TO_LABEL.get(pred_index, api_class_name or str(pred_index))
            st.success(f"Predicted class: **{friendly}**")

            # Probabilities handling
            if "probabilities" in out and out["probabilities"] is not None:
                probs = out["probabilities"]
                st.markdown("### Probabilities")
                if isinstance(probs, dict):
                    proba_df = pd.DataFrame.from_dict(probs, orient="index", columns=["probability"])
                    proba_df = proba_df.reset_index().rename(columns={"index":"class"})
                    # map class_x -> friendly label if possible
                    def _map_class(c):
                        try:
                            idx = int(str(c).split("_")[-1])
                            return CLASS_INDEX_TO_LABEL.get(idx, c)
                        except:
                            return c
                    proba_df["label"] = proba_df["class"].apply(_map_class)
                    proba_df = proba_df.set_index("label")
                    st.bar_chart(proba_df["probability"])
                    st.table(proba_df.reset_index().rename(columns={"index":"label", "probability":"prob"}))
                elif isinstance(probs, list):
                    proba_df = pd.DataFrame({
                        "label": [CLASS_INDEX_TO_LABEL.get(i, f"class_{i}") for i in range(len(probs))],
                        "probability": probs
                    }).set_index("label")
                    st.bar_chart(proba_df["probability"])
                    st.table(proba_df.reset_index())
                else:
                    st.write("Probabilities in unexpected format:", probs)

        except Exception as e:
            LOGGER.error("Error processing response", e)
            result_container.error("Failed to process server response. See logs.")

    # Call endpoints based on user action
    try:
        if predict_btn:
            if payload is None:
                result_container.error("No input provided.")
            else:
                # single predict (no probabilities expected)
                with st.spinner("Calling /predict ..."):
                    resp, elapsed = call_api("/predict", payload if isinstance(payload, dict) else {"input": payload})
                process_response(resp, elapsed)

        if proba_btn:
            if payload is None:
                result_container.error("No input provided.")
            else:
                # If payload is a list: try batch request with automatic fallback
                if isinstance(payload, list):
                    with st.spinner("Calling /predict-with-probability for batch (trying batch then fallback)..."):
                        ok, result_or_err, elapsed_total = try_batch_with_fallback("/predict-with-probability", payload)
                    if ok:
                        st.success(f"Batch prediction returned (elapsed {elapsed_total:.3f}s)")
                        # result_or_err may be a list of item responses or a batch response
                        st.write(result_or_err)
                    else:
                        result_container.error(f"Batch request failed: {result_or_err}")
                else:
                    # Single record
                    with st.spinner("Calling /predict-with-probability ..."):
                        resp, elapsed = call_api("/predict-with-probability", payload)
                    process_response(resp, elapsed)


        if batch_btn:
            uploaded_json = st.session_state.get("UPLOADED_JSON")
            if not uploaded_json:
                result_container.error("Please upload a JSON file containing a list of records for batch prediction.")
            else:
                # normalize and ensure list
                try:
                    normalized = normalize_uploaded_json(uploaded_json)
                except Exception as e:
                    result_container.error(str(e))
                    normalized = None

                if normalized is None:
                    pass
                elif not isinstance(normalized, list):
                    result_container.error("Uploaded JSON is not a list. For batch prediction upload JSON with a list of records (or input_batch).")
                else:
                    with st.spinner("Calling /predict-with-probability for batch (trying batch then fallback)..."):
                        ok, result_or_err, elapsed_total = try_batch_with_fallback("/predict-with-probability", normalized)
                    if ok:
                        st.success(f"Batch prediction returned (elapsed {elapsed_total:.3f}s)")
                        st.write(result_or_err)
                    else:
                        result_container.error(f"Batch request failed: {result_or_err}")


    except requests.exceptions.RequestException as e:
        LOGGER.error("Request failed", e)
        result_container.error("Request to backend failed. Check backend status and CORS settings.")

if __name__ == "__main__":
    run()
