from fastapi.testclient import TestClient

import main


async def fake_analyze_crop_anomaly(**_kwargs):
    return main.RecommendationResponse(
        assessment_text="Likely fungal leaf spot",
        anomaly_status=main.AnomalyStatus(
            is_real_problem=True,
            short_explanation="Visible lesions on leaves",
        ),
        top_suggestions=[
            main.Suggestion(
                rank=1,
                issue_name="Leaf spot",
                certainty="high",
                recommendation=main.Recommendation(
                    action_text="Remove infected leaves and spray fungicide",
                    chemical_name="Copper hydroxide",
                    urgency="soon",
                ),
            )
        ],
        monitoring_note="Recheck in 3 days.",
    )


def fake_get_openai_client():
    return object()


def run() -> None:
    main.analyze_crop_anomaly = fake_analyze_crop_anomaly
    main.get_openai_client = fake_get_openai_client

    with TestClient(main.app) as client:
        with open("Wheat_in_Dobritch.png", "rb") as image_file:
            response = client.post(
                "/v1/recommendation",
                data={"region": "Dobrich", "country": "Bulgaria", "region_type": "field"},
                files={"image": ("Wheat_in_Dobritch.png", image_file, "image/png")},
            )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["recommendation"]["top_suggestions"][0]["certainty"] == "high"
    print("smoke_test passed")


if __name__ == "__main__":
    run()
