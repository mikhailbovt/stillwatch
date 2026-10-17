import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from stillwatch.decisions import OpenAIDecisionProvider, parse_response, request_payload
from stillwatch.types import Category, EvidenceImage, EvidencePacket, Status


def packet(instruction_version="visible-evidence-v2"):
    return EvidencePacket(
        "a" * 32,
        "b" * 32,
        1,
        Category.MEDICAL,
        (EvidenceImage(1, 2.5, "overview", b"jpeg", 640, 480),),
        "rapid_posture_change",
        "config",
        2.5,
        instruction_version=instruction_version,
    )


def response(medical="REVIEW", fire="NO_ALERT"):
    return {
        "model": "gpt-6-luna",
        "answers": [
            {"name": "medical_status", "type": "choice", "choice": medical, "confidence": 0.7},
            {"name": "fire_status", "type": "choice", "choice": fire, "confidence": 0.8},
        ],
        "usage": {"input_tokens": 1234},
    }


@pytest.mark.parametrize("status", ["NO_ALERT", "REVIEW", "URGENT", "UNKNOWN"])
def test_valid_independent_answers(status):
    result = parse_response(packet(), response(status, "UNKNOWN"), "req-1")
    assert result.medical_status == Status(status) and result.fire_status == Status.UNKNOWN
    assert result.input_tokens == 1234 and result.request_id == "req-1"


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"model": "other", "answers": []},
        {
            "model": "gpt-6-luna",
            "answers": [{"name": "medical_status", "type": "choice", "choice": "invented"}],
        },
        {
            "model": "gpt-6-luna",
            "answers": [{"name": "medical_status", "type": "choice", "choice": "NO_ALERT"}],
        },
    ],
)
def test_invalid_and_partial_responses_are_unknown(raw):
    result = parse_response(packet(), raw)
    assert result.medical_status == result.fire_status == Status.UNKNOWN
    assert result.error


def test_refusal_does_not_erase_other_direction():
    raw = response()
    raw["answers"][0] = {"name": "medical_status", "type": "refusal"}
    result = parse_response(packet(), raw)
    assert result.medical_status == Status.UNKNOWN and result.fire_status == Status.NO_ALERT
    assert result.error == "provider_refusal"


def test_payload_has_typed_images_and_chronological_labels():
    body = request_payload(packet())
    assert body["model"] == "gpt-6-luna"
    assert len(body["questions"]) == 2
    assert body["input"][0]["content"][-1]["image_url"].startswith("data:image/jpeg;base64,")
    assert "2.500s" in body["input"][0]["content"][-2]["text"]


def test_saved_instruction_versions_keep_their_original_payload():
    current = request_payload(packet())
    previous = request_payload(replace(packet(), instruction_version="visible-evidence-v1"))
    assert "latest visible condition" in current["questions"][0]["instructions"]
    assert "latest visible condition" not in previous["questions"][0]["instructions"]
    assert current["questions"][1] == previous["questions"][1]
    assert current["input"] == previous["input"]


def test_unknown_instruction_version_is_rejected_before_submission():
    with pytest.raises(ValueError, match="Unsupported instruction version"):
        request_payload(replace(packet(), instruction_version="unknown"))


@pytest.mark.parametrize(
    "medical,observation,expected,reason",
    [
        ("NO_ALERT", "recovered", "NO_ALERT", None),
        ("NO_ALERT", "no_concern", "NO_ALERT", None),
        ("NO_ALERT", "view_lost", "UNKNOWN", "visible_outcome_uncertain"),
        ("NO_ALERT", "uncertain", "UNKNOWN", "visible_outcome_uncertain"),
        ("REVIEW", "view_lost", "UNKNOWN", "visible_outcome_uncertain"),
        ("URGENT", "uncertain", "UNKNOWN", "visible_outcome_uncertain"),
        ("REVIEW", "recovered", "UNKNOWN", "conflicting_medical_answers"),
        ("URGENT", "no_concern", "UNKNOWN", "conflicting_medical_answers"),
        ("NO_ALERT", "persistent_concern", "UNKNOWN", "no_alert_requires_visible_resolution"),
        ("URGENT", "persistent_concern", "REVIEW", "urgency_requires_visible_danger"),
        ("URGENT", "immediate_danger", "URGENT", None),
        ("REVIEW", "persistent_concern", "REVIEW", None),
    ],
)
def test_temporal_assessment_requires_a_consistent_visible_outcome(medical, observation, expected, reason):
    raw = response(medical, "URGENT")
    raw["answers"].append({"name": "medical_observation", "type": "choice", "choice": observation})
    result = parse_response(packet("visible-evidence-v3"), raw)
    assert result.error is None
    assert result.medical_status == expected and result.fire_status == Status.URGENT
    assert result.medical_observation == observation and result.policy_reason == reason
    assert result.raw_answer["answers"][0]["choice"] == medical


def test_temporal_payload_labels_observed_time_and_requires_outcome():
    p = replace(packet("visible-evidence-v3"), temporal_context={"elapsed_source_seconds": 15})
    body = request_payload(p)
    assert len(body["questions"]) == 3
    assert "elapsed_source_seconds" in body["input"][0]["content"][0]["text"]
    assert parse_response(p, response()).error == "incomplete_response"


def test_refused_outcome_cannot_confirm_or_clear_medical_concern():
    for status in ("NO_ALERT", "REVIEW", "URGENT"):
        raw = response(status, "URGENT")
        raw["answers"].append({"name": "medical_observation", "type": "refusal"})
        result = parse_response(packet("visible-evidence-v3"), raw)
        assert result.medical_status == Status.UNKNOWN
        assert result.fire_status == Status.URGENT
        assert result.error == "provider_refusal"
        assert result.policy_reason == "visible_outcome_unavailable"


def test_partial_recovery_instructions_preserve_the_saved_v3_payload():
    current = request_payload(packet("visible-evidence-v4"))
    previous = request_payload(packet("visible-evidence-v3"))
    assert "stably upright" in current["questions"][0]["instructions"]
    assert "stably upright" not in previous["questions"][0]["instructions"]
    assert "Sitting up" in current["questions"][2]["choices"][0]["description"]
    assert current["questions"][1] == previous["questions"][1]
    raw = response("REVIEW")
    raw["answers"].append({"name": "medical_observation", "type": "choice", "choice": "persistent_concern"})
    result = parse_response(packet("visible-evidence-v4"), raw)
    assert result.medical_status == Status.REVIEW and result.error is None


@pytest.mark.parametrize("posture", ["low", "seated_on_furniture", "unseen", "uncertain", "upright", None])
def test_recovery_clearance_requires_compatible_latest_posture(posture):
    raw = response("NO_ALERT", "URGENT")
    raw["answers"].append({"name": "medical_observation", "type": "choice", "choice": "recovered"})
    raw["answers"].append(
        {"name": "latest_posture", "type": "choice", "choice": posture}
        if posture is not None
        else {"name": "latest_posture", "type": "refusal"}
    )
    result = parse_response(packet("visible-evidence-v5"), raw)
    assert result.fire_status == Status.URGENT
    assert result.medical_status == (Status.NO_ALERT if posture == "upright" else Status.UNKNOWN)
    assert result.latest_posture == posture
    assert result.raw_answer == raw
    if posture != "upright":
        assert result.policy_reason == "recovery_not_supported_by_latest_posture"


def test_ordinary_rest_is_not_required_to_be_upright():
    raw = response("NO_ALERT")
    raw["answers"].extend(
        [
            {"name": "medical_observation", "type": "choice", "choice": "no_concern"},
            {"name": "latest_posture", "type": "choice", "choice": "low"},
        ]
    )
    result = parse_response(packet("visible-evidence-v5"), raw)
    assert result.medical_status == Status.NO_ALERT and result.policy_reason is None


def test_v5_adds_posture_without_rewriting_v4():
    previous = request_payload(packet("visible-evidence-v4"))
    current = request_payload(packet("visible-evidence-v5"))
    assert current["input"] == previous["input"]
    assert current["questions"][:3] == previous["questions"]
    assert len(current["questions"]) == 4
    assert current["questions"][-1]["name"] == "latest_posture"
    raw = response("NO_ALERT")
    raw["answers"].append({"name": "medical_observation", "type": "choice", "choice": "recovered"})
    assert parse_response(packet("visible-evidence-v4"), raw).medical_status == Status.NO_ALERT
    assert parse_response(packet("visible-evidence-v5"), raw).medical_status == Status.UNKNOWN


@pytest.mark.parametrize("status,retryable", [(401, False), (403, False), (429, True), (500, True)])
async def test_sdk_errors_are_redacted_and_classified(status, retryable):
    class Failure(Exception):
        status_code = status

    async def create(**kwargs):
        raise Failure("SECRET MUST NEVER ENTER LOGS")

    client = SimpleNamespace(decisions=SimpleNamespace(create=create))
    result = await OpenAIDecisionProvider(client=client).evaluate(packet())
    assert result.retryable == retryable
    assert "SECRET" not in result.model_dump_json()
    assert result.medical_status == Status.UNAVAILABLE


async def test_sdk_low_level_beta_path():
    class Client:
        async def post(self, path, cast_to, body):
            assert path == "/decisions" and body["questions"][0]["type"] == "choice"
            return httpx.Response(200, headers={"x-request-id": "beta-request"}, json=response())

    result = await OpenAIDecisionProvider(client=Client()).evaluate(packet())
    assert result.request_id == "beta-request" and result.input_tokens == 1234


async def test_corrupt_json_and_timeout_are_uncertain():
    class Corrupt:
        async def post(self, *args, **kwargs):
            return httpx.Response(200, text="broken")

    result = await OpenAIDecisionProvider(client=Corrupt()).evaluate(packet())
    assert result.error == json.JSONDecodeError.__name__ and result.medical_status != Status.NO_ALERT

    async def timeout(**kwargs):
        raise TimeoutError()

    result = await OpenAIDecisionProvider(
        client=SimpleNamespace(decisions=SimpleNamespace(create=timeout))
    ).evaluate(packet())
    assert result.retryable and result.medical_status == Status.UNAVAILABLE
