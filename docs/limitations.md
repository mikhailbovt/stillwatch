# Limitations

Stillwatch 0.5.0 is a preview for human review. It flags visible concerns and preserves evidence. It does not determine injury, medical safety or whether assistance is actually needed.

## Detection and context

- Small or distant people can be missed. Image size, camera angle, lighting and compression affect detection.
- Crowds, partial bodies and occlusion can fragment tracks or associate observations with the wrong person.
- Falls, controlled reclining, bending and slow loss of support are not reliably distinguished in every scene.
- Geometric recovery evidence can remain uncertain after a person is lost or obscured. A local recovery state is not proof of safety.
- Smoke, flame, reflections and lighting effects can be confused. Independent deployment accuracy remains unverified.
- A person already lying down at startup has no observed transition history. Missing evidence cannot establish a normal outcome.

## Assessment

Full medical assessment normally waits fifteen observed source seconds and twelve new frames. Capture, inference, queueing and provider latency add to that delay. An early screen has a separate immediate-danger path. The observation interval is an engineering default, not a clinically validated response time.

Cloud answers can be incorrect or unavailable. Medical clearance requires compatible local evidence. An established review remains visible through uncertain rechecks. `UNKNOWN`, missing models and stale frames are operationally distinct from `NO_ALERT`.

The default configuration uses mock answers and performs no semantic cloud assessment. Real assessment requires explicit cloud configuration, account access and spending limits. Only selected masked frames leave the server; see [deployment](deployment.md).

## Deployment

Use one process and one data directory per source. Actual analysis frequency depends on hardware, frame size and scene complexity. Four inference workers are not a process-wide CPU quota.

Independent false-alert rates, event recall and assistance classification on target cameras have not been established. Physical camera/RTSP installations and prolonged unattended operation require local validation. Development clips and repeated playback do not establish independent camera-hours.

Keep a human responsible for reviewing events and responding. Evaluate representative footage from the intended scene before relying on alerts.
