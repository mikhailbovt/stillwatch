# Third-party notices

Original Stillwatch service and SDK code: [Apache-2.0](LICENSE). Dependencies and model artifacts retain their own terms. The application license does not relicense those components.

| Component | Declared terms | Reference |
| --- | --- | --- |
| Ultralytics runtime and YOLO26 pose candidates | AGPL-3.0 / upstream commercial options | [Ultralytics licensing](https://www.ultralytics.com/license) |
| rabahdev Fire/Smoke YOLOv8n candidate | AGPL-3.0 as declared by its model card | [Pinned model](https://huggingface.co/rabahdev/fire-smoke-yolov8n/tree/13017fe8af477c25f5298d168e2dfede4b000753) |

The fire model author identifies D-Fire as training data. Source-data rights and model terms remain separate from the application license.

`uv.lock` identifies exact runtime distributions; their distribution metadata contains corresponding license notices. `models.lock.json` records model sources, revisions, hashes, classes and license references.

No third-party weights, source videos, annotations or event evidence are distributed. Numeric test fixtures preserve regression inputs and their provenance; they contain no video pixels. Documentation screenshots show the application without source footage.

Model conversion does not remove upstream terms. Review suitable model/runtime licensing for the intended deployment or supply a separately licensed perception adapter. The HTTP SDK has no inference dependency.
