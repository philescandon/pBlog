# Edge Inference Architecture Guide
## Integrating Tactical Datalinks with Lattice-Style Mesh Networks

*Generated: December 2024*

---

## Executive Summary

This document explores architectural options for integrating traditional tactical datalink systems (like L3Harris ADLT/RFE) with modern gossip-based mesh networks (like Anduril's Lattice). The key decision point is **where to perform inference** — onboard the aircraft, at the ground station, or both.

---

## Table of Contents

1. [Current Two-Box Architecture](#current-two-box-architecture)
2. [Integration Options](#integration-options)
3. [Compute Hardware for Edge Inference](#compute-hardware-for-edge-inference)
4. [Model Selection](#model-selection)
5. [Dual Inference Architecture](#dual-inference-architecture)
6. [Implementation Considerations](#implementation-considerations)
7. [Comparison Matrix](#comparison-matrix)

---

## Current Two-Box Architecture

The traditional tactical datalink follows a two-box design separating data processing from RF transmission.

### Airborne Segment

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AIRBORNE SEGMENT                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │   Sensors       │───▶│  ADLT           │───▶│  RFE           │  │
│  │ (Camera, EO/IR) │    │ (Air Data Link  │    │ (Radio Freq    │──┼──► RF Out
│  │                 │    │  Terminal)      │    │  Electronics)  │  │
│  └─────────────────┘    └─────────────────┘    └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**ADLT (Air Data Link Terminal)**
- Handles encoding, encryption, multiplexing of sensor data
- Manages the protocol stack
- Receives command/control uplink

**RFE (Radio Frequency Electronics)**
- Converts digital to RF and vice versa
- Handles modulation/demodulation
- Power amplification for transmission
- May include frequency hopping, spread spectrum

### Ground Segment

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GROUND SEGMENT                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │  GDT Antenna    │───▶│  Ground RFE     │───▶│  Ground        │  │
│  │  (Tracking)     │    │  (Receiver)     │    │  Terminal/GCS  │  │
│  └─────────────────┘    └─────────────────┘    └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Characteristics

| Direction | Bandwidth | Content |
|-----------|-----------|---------|
| **Downlink** (Air → Ground) | 10.71 - 274 Mbps | Full-motion video, imagery, sensor metadata |
| **Uplink** (Ground → Air) | 200 Kbps - 2 Mbps | Flight commands, payload control, mission updates |

---

## Integration Options

### Option A: Ground-Only Inference

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  UAV        │      │  Ground     │      │  Lattice    │
│  Sensors ──▶│ ADLT │──▶ Terminal │──▶   │  Mesh       │
│  (Raw video)│      │  + Your     │      │  (Entities) │
│             │      │  Inference  │      │             │
└─────────────┘      └─────────────┘      └─────────────┘
```

**Advantages:**
- Full control over inference pipeline
- Unlimited compute resources
- Easy model updates
- Full video archive retained
- Can run multiple/ensemble models

**Disadvantages:**
- Higher latency (network transit time)
- Depends on ground station availability
- Full bandwidth required for video downlink

### Option B: Airborne-Only Inference

```
┌─────────────┐      ┌─────────────┐
│  UAV        │      │  Lattice    │
│  Sensors ──▶│ ADLT │──▶ Mesh     │
│  + Onboard  │      │  (Entities) │
│  Inference  │      │             │
└─────────────┘      └─────────────┘
```

**Advantages:**
- Lowest latency (milliseconds)
- Works without ground station
- Lower bandwidth (entities vs video)
- Survivable if ground is compromised

**Disadvantages:**
- Limited by airborne SWaP (Size, Weight, Power)
- Harder to update models
- May lose raw video data
- Smaller models = potentially lower accuracy

### Option C: Dual Inference (Recommended)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AIRBORNE SEGMENT                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐          │
│  │ Sensor   │───▶│ Onboard  │───▶│  ADLT    │───▶│ RFE  │──► RF    │
│  │ (EO/IR)  │    │ Compute  │    │          │    │      │          │
│  └──────────┘    │ (YOLOv8) │    │ Sends:   │    └──────┘          │
│                  └────┬─────┘    │ -Entities│                      │
│                       │          │ -Video   │                      │
│              Detections ready    └──────────┘                      │
│              in ~30ms                                              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ───────────────┼───────────────
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GROUND SEGMENT                              │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐                 │
│  │ GDT      │───▶│ Ground   │───▶│ Ground        │                 │
│  │ Antenna  │    │ RFE      │    │ Terminal      │                 │
│  └──────────┘    └──────────┘    └───────┬───────┘                 │
│                                          │                         │
│                          ┌───────────────┼───────────────┐         │
│                          ▼               ▼               ▼         │
│                   ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│                   │ Video    │    │ Ground   │    │ Airborne │     │
│                   │ Archive  │    │ Inference│    │ Entities │     │
│                   │          │    │ (YOLOv8m)│    │ Received │     │
│                   └──────────┘    └────┬─────┘    └────┬─────┘     │
│                                        │               │           │
│                                        ▼               ▼           │
│                                   ┌────────────────────────┐       │
│                                   │   Detection Fusion     │       │
│                                   │   & Verification       │       │
│                                   └───────────┬────────────┘       │
│                                               │                    │
└───────────────────────────────────────────────┼────────────────────┘
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │      Lattice Mesh      │
                                   │   (Gossip Protocol)    │
                                   │                        │
                                   │  - Fused entities      │
                                   │  - High confidence     │
                                   │  - Your classification │
                                   └────────────────────────┘
```

**Advantages:**
- Immediate tactical response (airborne inference)
- Verification and higher accuracy (ground inference)
- Agreement between systems = higher confidence
- Full data retention
- Flexibility to run experimental models

**Disadvantages:**
- More complex system
- Higher cost (compute on both ends)
- Need fusion logic

---

## Compute Hardware for Edge Inference

### Airborne Compute Options

| Hardware | Power Draw | AI Performance | Form Factor | Approx. Cost | Best For |
|----------|------------|----------------|-------------|--------------|----------|
| **NVIDIA Jetson AGX Orin** | 15-60W | 275 TOPS (INT8) | 100x87mm | $1,999 | High-end UAV, maximum accuracy |
| **NVIDIA Jetson Orin NX** | 10-25W | 100 TOPS (INT8) | 69.6x45mm | $599 | Mid-size UAV, good balance |
| **NVIDIA Jetson Orin Nano** | 7-15W | 40 TOPS (INT8) | 69.6x45mm | $249 | Smaller platforms |
| **Hailo-8** | 2.5W | 26 TOPS | M.2 module | ~$100 | Extreme SWaP constraints |
| **Hailo-8L** | 1.5W | 13 TOPS | M.2 module | ~$70 | Smallest drones |
| **Intel Movidius Myriad X** | 1-2W | 4 TOPS | USB stick | ~$80 | Legacy/simple detection |
| **Xilinx Kria K26** | 6-10W | Custom | SOM | $349 | Mil-spec, customizable |
| **AMD/Xilinx Versal AI Edge** | 10-35W | 100+ TOPS | Various | $1,000+ | Rad-hardened options available |

### Ground Compute Options

| Hardware | Power Draw | AI Performance | Approx. Cost | Best For |
|----------|------------|----------------|--------------|----------|
| **NVIDIA RTX 4090** | 450W | 1,321 TOPS (INT8) | $1,599 | Maximum throughput |
| **NVIDIA RTX 4080** | 320W | 780 TOPS (INT8) | $1,199 | High performance |
| **NVIDIA L4** | 72W | 485 TOPS (INT8) | $2,500 | Data center/rack mount |
| **NVIDIA A4000** | 140W | 153 TOPS | $1,000 | Workstation |
| **Apple M3 Max** | 40-90W | ~18 TOPS (Neural Engine) | $3,199+ | MacBook-based dev |
| **Apple M3 Ultra** | ~100W | ~36 TOPS (Neural Engine) | $4,999+ | Mac Studio |

### Recommended Configurations

**Tight SWaP Budget (Small UAV):**
```
Airborne: Hailo-8 (2.5W, 26 TOPS)
Model: YOLOv8n quantized to INT8
Expected: 40-60 FPS @ 640x640
```

**Balanced (Medium UAV):**
```
Airborne: Jetson Orin NX (15W mode)
Model: YOLOv8s FP16
Expected: 45-60 FPS @ 640x640
```

**Performance Priority (Large UAV/High Value):**
```
Airborne: Jetson AGX Orin (30W mode)
Model: YOLOv8m FP16 or ensemble
Expected: 30-45 FPS @ 1280x1280
```

---

## Model Selection

### YOLO Version Comparison

| Model | Parameters | mAP@50-95 | Inference (Orin NX) | Recommendation |
|-------|------------|-----------|---------------------|----------------|
| YOLOv8n | 3.2M | 37.3 | 60+ FPS | Extreme SWaP constraints |
| YOLOv8s | 11.2M | 44.9 | 45 FPS | **Airborne (recommended)** |
| YOLOv8m | 25.9M | 50.2 | 30 FPS | Ground or high-power airborne |
| YOLOv8l | 43.7M | 52.9 | 18 FPS | Ground only |
| YOLOv8x | 68.2M | 53.9 | 12 FPS | Ground, maximum accuracy |

### Model Optimization Techniques

**Quantization:**
```python
# Convert FP32 model to INT8 for edge deployment
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format="engine", int8=True, data="calibration_data.yaml")
```

**TensorRT Optimization:**
```python
# Export to TensorRT for NVIDIA hardware
model.export(format="engine", device=0, half=True)  # FP16
```

**ONNX for Cross-Platform:**
```python
model.export(format="onnx", simplify=True, opset=12)
```

### Why Not Use Lattice's YOLOv4?

The Lattice documentation references YOLOv4, but for new development:

| Factor | YOLOv4 | YOLOv8 |
|--------|--------|--------|
| Architecture | Anchor-based | Anchor-free |
| Training | Complex config | Simple API |
| Performance | Good | Better at same compute |
| Ecosystem | Mature but aging | Active development |
| Recommendation | Legacy compatibility | **New projects** |

---

## Dual Inference Architecture

### Detection Fusion Logic

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    source: str  # "airborne" | "ground" | "fused"
    timestamp: float
    track_id: Optional[int] = None

class DualInferenceFusion:
    """
    Fuse airborne and ground detections for higher confidence output.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def find_match(self, detection: Detection,
                   candidates: List[Detection]) -> Optional[Detection]:
        """Find matching detection based on IoU threshold."""
        best_match = None
        best_iou = self.iou_threshold

        for candidate in candidates:
            iou = self.compute_iou(detection.bbox, candidate.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = candidate

        return best_match

    def resolve_class(self, det1: Detection, det2: Detection) -> str:
        """
        Resolve class disagreement between two detections.
        Prefer higher confidence, or ground if close.
        """
        if det1.class_name == det2.class_name:
            return det1.class_name

        # If confidence difference > 20%, trust higher confidence
        if abs(det1.confidence - det2.confidence) > 0.2:
            return det1.class_name if det1.confidence > det2.confidence else det2.class_name

        # Otherwise prefer ground (larger model, more accurate)
        return det2.class_name if det2.source == "ground" else det1.class_name

    def fuse(self, airborne: List[Detection],
             ground: List[Detection]) -> List[Detection]:
        """
        Fuse airborne and ground detections.

        Strategy:
        - Matched detections: Boost confidence, use ground bbox
        - Airborne-only: Include with slight confidence penalty
        - Ground-only: Include (airborne may have missed)
        """
        fused = []
        matched_ground = set()

        for air_det in airborne:
            ground_match = self.find_match(air_det, ground)

            if ground_match:
                # Both systems detected — high confidence
                matched_ground.add(id(ground_match))
                fused.append(Detection(
                    bbox=ground_match.bbox,  # Ground usually more precise
                    class_name=self.resolve_class(air_det, ground_match),
                    confidence=min(0.99, max(air_det.confidence, ground_match.confidence) * 1.1),
                    source="fused",
                    timestamp=air_det.timestamp,  # Use earlier timestamp
                    track_id=air_det.track_id or ground_match.track_id
                ))
            else:
                # Airborne-only — include but flag
                fused.append(Detection(
                    bbox=air_det.bbox,
                    class_name=air_det.class_name,
                    confidence=air_det.confidence * 0.85,  # Slight penalty
                    source="airborne",
                    timestamp=air_det.timestamp,
                    track_id=air_det.track_id
                ))

        # Add ground-only detections
        for g_det in ground:
            if id(g_det) not in matched_ground:
                fused.append(Detection(
                    bbox=g_det.bbox,
                    class_name=g_det.class_name,
                    confidence=g_det.confidence * 0.9,  # Slight penalty for no airborne confirm
                    source="ground",
                    timestamp=g_det.timestamp,
                    track_id=g_det.track_id
                ))

        return fused
```

### Entity Publishing to Mesh

```python
import zmq
import json
from datetime import datetime

class LatticePublisher:
    """
    Publish fused detections to Lattice-style mesh network.
    """

    def __init__(self, node_id: str, pub_address: str = "tcp://*:5555"):
        self.node_id = node_id
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(pub_address)

    def detection_to_entity(self, detection: Detection,
                           geo_location: tuple) -> dict:
        """
        Convert detection to Lattice entity format.
        """
        return {
            "entity_id": f"{self.node_id}_{detection.track_id or 'untracked'}",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "location": {
                    "lat": geo_location[0],
                    "lon": geo_location[1],
                    "alt": geo_location[2] if len(geo_location) > 2 else None,
                    "source": self.node_id
                },
                "classification": {
                    "type": detection.class_name,
                    "confidence": detection.confidence,
                    "model": "yolov8_dual_inference"
                },
                "detection_meta": {
                    "source": detection.source,
                    "bbox_image": detection.bbox,
                    "track_id": detection.track_id
                }
            },
            "origin_node": self.node_id,
            "hop_count": 0
        }

    def publish(self, detection: Detection, geo_location: tuple):
        """
        Publish entity to mesh with topic-based routing.
        """
        entity = self.detection_to_entity(detection, geo_location)
        topic = f"track/{detection.class_name}"

        self.publisher.send_multipart([
            topic.encode(),
            json.dumps(entity).encode()
        ])
```

---

## Implementation Considerations

### Latency Budget

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LATENCY BREAKDOWN                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  AIRBORNE PATH (Option B):                                         │
│  ├─ Sensor capture:           ~5 ms                                │
│  ├─ Onboard inference:       ~30 ms (YOLOv8s @ 30 FPS)            │
│  ├─ Entity encoding:          ~1 ms                                │
│  ├─ RF transmission:         ~10 ms                                │
│  └─ Total to mesh:          ~46 ms                                 │
│                                                                     │
│  GROUND PATH (Option A):                                           │
│  ├─ Sensor capture:           ~5 ms                                │
│  ├─ Video encoding:          ~10 ms                                │
│  ├─ RF transmission:         ~10 ms                                │
│  ├─ Video decoding:           ~5 ms                                │
│  ├─ Ground inference:        ~20 ms (YOLOv8m on RTX)              │
│  ├─ Entity encoding:          ~1 ms                                │
│  └─ Total to mesh:          ~51 ms                                 │
│                                                                     │
│  DUAL PATH (Option C):                                             │
│  ├─ Airborne entities:       ~46 ms (immediate publish)            │
│  ├─ Ground verification:     ~51 ms (may update/override)          │
│  └─ Fused high-confidence:   ~55 ms                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Bandwidth Considerations

| Data Type | Size | Notes |
|-----------|------|-------|
| Raw video frame (1080p) | ~6 MB | Uncompressed |
| H.264 encoded frame | ~50-200 KB | Typical compression |
| H.265/HEVC frame | ~30-150 KB | Better compression |
| Single entity (JSON) | ~500 bytes | Detection only |
| Entity with metadata | ~2 KB | Full component set |

**Key insight**: Sending entities instead of video reduces bandwidth by **100-1000x**.

### Failure Mode Handling

```python
class ResilientPublisher:
    """
    Handle network failures gracefully.
    """

    def __init__(self):
        self.local_buffer = []  # Store if mesh unavailable
        self.max_buffer = 1000

    def publish_with_fallback(self, entity: dict, mesh_available: bool):
        if mesh_available:
            # Flush buffer first
            for buffered in self.local_buffer:
                self._send_to_mesh(buffered)
            self.local_buffer.clear()

            # Send current
            self._send_to_mesh(entity)
        else:
            # Buffer locally
            if len(self.local_buffer) < self.max_buffer:
                self.local_buffer.append(entity)
            else:
                # Drop oldest
                self.local_buffer.pop(0)
                self.local_buffer.append(entity)
```

---

## Comparison Matrix

### Architecture Options Summary

| Factor | Ground Only | Airborne Only | Dual Inference |
|--------|-------------|---------------|----------------|
| **Latency to mesh** | ~51 ms | ~46 ms | ~46 ms (air) / ~55 ms (fused) |
| **Accuracy** | Highest | Limited by SWaP | High (verified) |
| **Bandwidth required** | High (video) | Low (entities) | High (video + entities) |
| **Ground station dependency** | Critical | None | Partial |
| **Model flexibility** | Maximum | Constrained | Both |
| **Data retention** | Full video | Entities only | Full video |
| **System complexity** | Low | Low | Medium |
| **Cost** | Low | Medium | Higher |
| **Recommended for** | Development, high-value targets | Contested comms, swarms | Production systems |

### Decision Framework

```
START
  │
  ▼
Is ground station reliable and available?
  │
  ├─ NO ──► Use Airborne-Only (Option B)
  │
  ▼ YES
  │
Is latency critical (< 50ms requirement)?
  │
  ├─ YES ──► Use Airborne-Only (Option B) or Dual with airborne priority
  │
  ▼ NO
  │
Do you need full video archive?
  │
  ├─ YES ──► Use Dual Inference (Option C) or Ground-Only (Option A)
  │
  ▼ NO
  │
Is airborne SWaP available for compute?
  │
  ├─ NO ──► Use Ground-Only (Option A)
  │
  ▼ YES
  │
Use Dual Inference (Option C) ◄── RECOMMENDED DEFAULT
```

---

## Appendix: Quick Start Code

### Airborne Inference Setup (Jetson)

```python
# airborne_inference.py
from ultralytics import YOLO
import cv2

# Load optimized model
model = YOLO("yolov8s.engine")  # TensorRT optimized

cap = cv2.VideoCapture(0)  # Sensor input

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.track(frame, persist=True, verbose=False)

    # Extract detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Package as entity and send via ADLT
            entity = {
                "bbox": box.xyxy[0].tolist(),
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "track_id": int(box.id) if box.id else None
            }
            send_to_adlt(entity)
```

### Ground Fusion Setup

```python
# ground_fusion.py
from dual_inference_fusion import DualInferenceFusion, Detection

fusion = DualInferenceFusion(iou_threshold=0.5)

def process_frame(video_frame, airborne_entities):
    # Run ground inference
    ground_results = ground_model.track(video_frame, persist=True)
    ground_detections = [Detection(...) for r in ground_results]

    # Convert airborne entities to Detection objects
    airborne_detections = [Detection(...) for e in airborne_entities]

    # Fuse
    fused = fusion.fuse(airborne_detections, ground_detections)

    # Publish to Lattice mesh
    for det in fused:
        publisher.publish(det, geo_location)
```

---

## References

- Anduril Lattice OS Architecture Guide (internal document)
- NVIDIA Jetson Documentation: https://developer.nvidia.com/embedded/jetson-modules
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- ZeroMQ Guide: https://zguide.zeromq.org/
- Tactical Common Data Link (TCDL) specifications

---

*Document version 1.0 — For internal use*
