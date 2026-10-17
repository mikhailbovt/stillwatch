import asyncio
from datetime import UTC, datetime

import pytest

from stillwatch.decisions import MockDecisionProvider
from stillwatch.scheduler import Budget
from stillwatch.types import Box, Candidate, Category, DecisionResult, Status


def add(manager, x=0, category=Category.MEDICAL):
    return manager.ingest(
        Candidate(
            category=category,
            reason="persistent_low_posture",
            first_ts=6,
            last_ts=6,
            box=Box(x1=x, y1=0, x2=x + 30, y2=60),
            track_id=x + 1,
            epoch=1,
        )
    )[0]


async def test_minute_lying_episode_has_at_most_three_requests(system):
    _, _, _, _, manager, provider, scheduler = system
    item = add(manager)
    scheduler.enqueue(item)
    for ts in range(6, 67):
        item.last_ts = ts
        scheduler.advance(ts, 1)
        await scheduler.step()
    assert item.request_count == 3
    assert len(provider.calls) <= 4  # One independent startup overview.


async def test_overload_is_persisted_not_lost_and_categories_are_independent(system):
    settings, _, storage, _, manager, _, scheduler = system
    for index in range(100):
        item = add(manager, x=index * 100, category=Category.FIRE if index % 2 else Category.MEDICAL)
        scheduler.enqueue(item)
    assert len(scheduler.jobs) == settings.queue_capacity
    assert len(storage.incidents()) == 100
    assert any(i.queue_reason == "queue_capacity_wait" for i in manager.items.values())
    await scheduler.step()
    scheduler.advance(7, 1)
    assert len(scheduler.jobs) <= settings.queue_capacity


async def test_budget_exhaustion_does_not_hide_candidates(system):
    settings, _, storage, _, manager, _, scheduler = system
    settings.requests_per_minute = 1
    first = add(manager)
    scheduler.enqueue(first)
    await scheduler.step()
    second = add(manager, x=100, category=Category.FIRE)
    scheduler.enqueue(second)
    assert not await scheduler.step()
    assert second.queue_reason == "minute_request_limit"
    assert second.request_count == 0 and second.incident_id in scheduler.jobs
    assert len(storage.incidents()) == 2


async def test_stuck_api_is_bounded_and_closing_keeps_response_stale(system):
    _, _, storage, _, manager, _, scheduler = system
    scheduler.provider = MockDecisionProvider(delay=0.05, statuses=(Status.URGENT, Status.NO_ALERT))
    item = add(manager)
    scheduler.enqueue(item)
    task = asyncio.create_task(scheduler.step())
    await asyncio.sleep(0.01)
    manager.annotate(item.incident_id, "close")
    assert not await scheduler.step()
    await task
    assert item.analysis_status == Status.PENDING
    assert storage.detail(item.incident_id)["assessments"][0]["stale"]


async def test_mismatched_response_is_not_attached(system):
    _, _, _, _, manager, _, scheduler = system

    class Wrong:
        name = "mock"

        async def evaluate(self, packet):
            return DecisionResult(
                incident_id="other",
                packet_id=packet.packet_id,
                version=packet.version,
                medical_status=Status.NO_ALERT,
                fire_status=Status.NO_ALERT,
                provider="mock",
                model="wrong",
            )

    scheduler.provider = Wrong()
    item = add(manager)
    scheduler.enqueue(item)
    await scheduler.step()
    assert item.analysis_status == Status.UNKNOWN
    assert item.queue_reason == "response_binding_mismatch"


async def test_unavailable_source_preserves_queue_without_reserving_cloud_spend(system):
    _, _, _, _, manager, provider, scheduler = system
    item = add(manager)
    scheduler.enqueue(item)
    scheduler.can_send = lambda: False
    assert not await scheduler.step()
    assert scheduler.budget.snapshot()["requests"] == 0
    assert not provider.calls and item.queue_reason == "source_not_fresh"
    assert item.incident_id in scheduler.jobs
    scheduler.can_send = lambda: True
    assert await scheduler.step()


async def test_source_disconnect_during_request_keeps_result_in_history(system):
    _, _, storage, _, manager, _, scheduler = system
    fresh = True
    scheduler.can_send = lambda: fresh
    scheduler.provider = MockDecisionProvider(delay=0.05, statuses=(Status.NO_ALERT, Status.NO_ALERT))
    item = add(manager)
    scheduler.enqueue(item)
    task = asyncio.create_task(scheduler.step())
    await asyncio.sleep(0.01)
    fresh = False
    await task
    assert item.analysis_status == Status.PENDING and item.assessment_stale
    assert storage.detail(item.incident_id)["assessments"][0]["stale"]
    fresh = True
    scheduler.advance(7, 1)
    assert item.incident_id in scheduler.jobs


async def test_overview_is_independent_of_candidates(system):
    _, _, _, _, manager, _, scheduler = system
    scheduler.advance(6, 1)
    await scheduler.step()
    assert any(i.category == Category.OVERVIEW for i in manager.items.values())
    scheduler.advance(126, 1)
    assert len([i for i in manager.items.values() if i.category == Category.OVERVIEW]) == 2


def test_budget_survives_restart_and_utc_midnight(system):
    settings, _, storage, _, _, _, _ = system
    settings.requests_per_minute = 1
    budget = Budget(settings, storage, "openai")
    before = datetime(2026, 10, 15, 23, 59, 59, tzinfo=UTC).timestamp()
    assert budget.reserve(before)[0] is not None
    restored = Budget(settings, storage, "openai")
    assert restored.reserve(before + 2)[1] == "minute_request_limit"
    assert restored.snapshot(before + 2)["requests"] == 0
    assert restored.snapshot(before)["input_tokens"] == settings.token_reservation


async def test_source_loop_resets_geometry_without_resetting_overview_cadence(system):
    _, _, _, _, manager, _, scheduler = system
    scheduler.advance(6, 1)
    await scheduler.step()
    for epoch in range(2, 15):
        scheduler.advance(0, epoch)
        scheduler.advance(6, epoch)
    assert len([i for i in manager.items.values() if i.category == Category.OVERVIEW]) == 1
    scheduler.advance(60, 15)
    assert len([i for i in manager.items.values() if i.category == Category.OVERVIEW]) == 2


async def test_explicit_source_change_starts_a_fresh_overview_after_evidence(system):
    _, buffer, _, _, manager, provider, scheduler = system
    scheduler.advance(6, 1)
    await scheduler.step()
    scheduler.change_source(2)
    assert scheduler.last_overview_completed_mono is None
    assert not scheduler.jobs
    # Opening the new capture creates another epoch. No old frame is eligible.
    scheduler.advance(0, 3)
    assert not await scheduler.step()
    assert len(provider.calls) == 1
    assert scheduler.budget.snapshot()["requests"] == 1
    assert scheduler.budget.snapshot()["unknown_usage_requests"] == 0
    from dataclasses import replace

    frame = replace(buffer.frames()[-1], epoch=3, capture_ts=0, frame_id=100)
    buffer.clear()
    buffer.add(frame)
    assert await scheduler.step()
    assert len(provider.calls) == 2
    current = [i for i in manager.items.values() if i.epoch == 3]
    assert len(current) == 1 and current[0].request_count == 1
    assert scheduler.last_overview_completed_mono is not None


@pytest.mark.parametrize("status", [Status.UNKNOWN, Status.REVIEW, Status.URGENT, Status.NO_ALERT])
async def test_all_mock_statuses_are_distinct(system, status):
    _, _, _, evidence, manager, _, _ = system
    item = add(manager)
    result = await MockDecisionProvider(statuses=(status, Status.NO_ALERT)).evaluate(evidence.build(item, 6))
    assert result.medical_status == status
