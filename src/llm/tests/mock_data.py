from prompts import BASE_PROMPT, QUESTION_PROMPT

MOCK_PREDICTIONS = {"eczema": 0.78, "psoriasis": 0.15, "contact_dermatitis": 0.05, "seborrheic_dermatitis": 0.02}
MOCK_QUESTION1 = "How long does it take to heal?"
MOCK_QUESTION2 = "What should I avoid?"
MOCK_ANSWER1 = "It typically takes 2-4 weeks for the skin to heal, but it can vary depending on the severity of the condition."

MOCK_METADATA = {
    "user_input": "I've had a red, itchy patch on my elbow for two weeks. It gets worse with hot showers.",
    "cv_analysis": {"area": 8.4, "color_profile": {"average_Lab": [67.2, 18.4, 9.3], "redness_index": 0.34}},
    "history": {
        "2025-10-10": {"cv_analysis": {"area": 12.6, "color_profile": {"redness_index": 0.46}}, "text_summary": "Patient reported increased redness and itching."},
        "2025-11-01": {"cv_analysis": {"area": 9.8, "color_profile": {"redness_index": 0.38}}, "text_summary": "Slight improvement noted."}
    }
}