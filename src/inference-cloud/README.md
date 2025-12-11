# Inference Cloud Service

FastAPI service for skin disease classification using multimodal vision and text embeddings.

## Overview

Provides ML inference endpoints for:
- **Text embedding**: Generate embeddings from patient descriptions
- **Disease prediction**: Classify skin conditions from vision + text embeddings
- **Model info**: Get available disease classes

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health with model info
- `POST /embed-text` - Generate text embedding
- `POST /predict` - Predict skin condition from embeddings
- `GET /classes` - List all available disease classes

## Quick Start

### Local Development

```bash
# Build and run
docker build -f Dockerfile -t inference-cloud:latest ../..
docker run -p 8080:8080 inference-cloud:latest
```

### Environment Variables

```bash
export PORT=8080
export MODEL_CHECKPOINT_PATH=/tmp/models/test_best.pth
export MODEL_GCS_PATH=gs://apcomp215-datasets/test_best.pth
export DEVICE=cpu  # or cuda
```

### Run Tests

```bash
# Unit tests
pytest tests/test_unit.py -v

# Integration tests
pytest tests/test_integration.py -v

# System tests (requires deployed service)
pytest tests/test_system.py -v
```

## Architecture

- **FastAPI** application with async endpoints
- **MultimodalClassifier** from `ml_workflow` for predictions
- **Text encoder** (transformers) for text embeddings
- **Model loading** from GCS on startup (via `docker-entrypoint.sh`)

## Deployment

Deployed to GKE via Pulumi (see `src/deployment/modules/gke_inference.py`).

Service automatically downloads model checkpoint from GCS on startup if `MODEL_GCS_PATH` is set.

### What remains untested

<table class="index" data-sortable>
        <thead>
            <tr class="tablehead" title="Click to sort">
                <th id="file" class="name" aria-sort="none" data-shortcut="f">File<span class="arrows"></span></th>
                <th class="spacer">&nbsp;</th>
                <th id="statements" aria-sort="none" data-default-sort-order="descending" data-shortcut="s">statements<span class="arrows"></span></th>
                <th id="missing" aria-sort="none" data-default-sort-order="descending" data-shortcut="m">missing<span class="arrows"></span></th>
                <th id="excluded" aria-sort="none" data-default-sort-order="descending" data-shortcut="x">excluded<span class="arrows"></span></th>
                <th class="spacer">&nbsp;</th>
                <th id="coverage" aria-sort="none" data-shortcut="c">coverage<span class="arrows"></span></th>
            </tr>
        </thead>
        <tbody>
            <tr class="region">
                <td class="name"><a href="inference_classifier_py.html">inference_classifier.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>167</td>
                <td>47</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="120 167">72%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="main_py.html">main.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>86</td>
                <td>9</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="77 86">90%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_a44f0ac069e85531_test_system_py.html">tests&#8201;/&#8201;test_system.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>128</td>
                <td>15</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="113 128">88%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_c7f111f429cd1ffd_dataloader_py.html">&#8201;/&#8201;app&#8201;/&#8201;ml_workflow&#8201;/&#8201;dataloader&#8201;/&#8201;dataloader.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>185</td>
                <td>163</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="22 185">12%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_c7f111f429cd1ffd_embedding_utils_py.html">&#8201;/&#8201;app&#8201;/&#8201;ml_workflow&#8201;/&#8201;dataloader&#8201;/&#8201;embedding_utils.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>151</td>
                <td>124</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="27 151">18%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_c7f111f429cd1ffd_transform_utils_py.html">&#8201;/&#8201;app&#8201;/&#8201;ml_workflow&#8201;/&#8201;dataloader&#8201;/&#8201;transform_utils.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>66</td>
                <td>55</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="11 66">17%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_e08fa19da2d30d30_io_utils_py.html">&#8201;/&#8201;app&#8201;/&#8201;ml_workflow&#8201;/&#8201;io_utils.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>123</td>
                <td>111</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="12 123">10%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_fa7d6649aff8e1d8_multimodal_classifier_py.html">&#8201;/&#8201;app&#8201;/&#8201;ml_workflow&#8201;/&#8201;model&#8201;/&#8201;classifier&#8201;/&#8201;multimodal_classifier.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>169</td>
                <td>87</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="82 169">49%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_160e8fc51a4def9e_utils_py.html">&#8201;/&#8201;app&#8201;/&#8201;ml_workflow&#8201;/&#8201;model&#8201;/&#8201;utils.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>53</td>
                <td>24</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="29 53">55%</td>
            </tr>
            <tr class="region">
                <td class="name"><a href="z_e08fa19da2d30d30_utils_py.html">&#8201;/&#8201;app&#8201;/&#8201;ml_workflow&#8201;/&#8201;utils.py</a></td>
                <td class="spacer">&nbsp;</td>
                <td>265</td>
                <td>237</td>
                <td>0</td>
                <td class="spacer">&nbsp;</td>
                <td data-ratio="28 265">11%</td>
            </tr>
        </tbody>
    </table>
