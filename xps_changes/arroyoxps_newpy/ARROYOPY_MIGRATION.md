# Migrating AP-XPS to YAML-driven arroyopy Blocks

This document describes how to migrate from Python CLI-based operator wiring to the new
YAML-driven `arroyopy` block system.

---

## Overview

The new `arroyopy` introduces a `Block` abstraction and a YAML configuration loader.
Instead of writing a `*_cli.py` file that manually instantiates and wires together
operators, listeners, and publishers, you declare the pipeline in a YAML file and run it
with the `arroyo` CLI.

**Old approach:**
```python
# processor_cli_tpx.py
operator = XPSOperator()
ws_publisher = XPSWSResultPublisher(app_settings.websocket_url)
operator.add_publisher(ws_publisher)
tpx_zmq_socket = setup_zmq()
listener = XPSTimepixZMQListener(operator=operator, zmq_socket=tpx_zmq_socket)
await asyncio.gather(listener.start(), ws_publisher.start())
```

**New approach:**
```yaml
# pipelines/timepix_pipeline.yaml
blocks:
  - name: XPS Timepix Operator
    operator:
      class: tr_ap_xps.pipeline.xps_operator.XPSOperator
      kwargs:
        build_heatmaps: false
    listeners:
      - class: tr_ap_xps.timepix.XPSTimepixZMQListener
        kwargs:
          zmq_pub_address: "tcp://localhost"
          zmq_pub_port: 5657
    publishers:
      - class: tr_ap_xps.websockets.XPSWSResultPublisher
        kwargs:
          ws_url: "ws://0.0.0.0:8001/xps_operator"
```

```bash
arroyo run pipelines/timepix_pipeline.yaml
```

---

## Files Deleted

```
src/tr_ap_xps/apps/processor_cli.py       ← deleted
src/tr_ap_xps/apps/processor_cli_tpx.py   ← deleted
src/tr_ap_xps/apps/__init__.py             ← deleted
```

The `dynaconf`-based `settings.yaml` and `config.py` can also be removed since all
configuration now lives in the pipeline YAML files.

---

## Key arroyopy API Changes

| Area | Old | New |
|------|-----|-----|
| `Operator.__init__` | no `super().__init__()` needed | must call `super().__init__()` to set up `listener_queue` |
| `Operator.start()` | not present / overridden freely | base class runs a queue loop; **must override** in AP-XPS to avoid double-publish |
| `Listener.start()` | no fixed signature | `async def start(self) -> None` — no args |
| `Block` | not present | new container class; wires operator + listeners + publishers |
| YAML config | not supported | `load_blocks_from_yaml()` + `arroyo run` CLI |

---

## Required Code Changes

### 1. `xps_operator.py`

Add `super().__init__()` and override `start()` to prevent the base class queue loop
from double-publishing results (since `XPSOperator.process()` already calls
`self.publish()` internally).

```python
class XPSOperator(Operator):
    def __init__(self, build_heatmaps: bool = False) -> None:
        super().__init__()  # required — sets up listener_queue
        self.xps_processor = None
        self.build_heatmaps = build_heatmaps

    async def process(self, message: Message) -> None:
        # existing logic unchanged — calls self.publish(result) internally
        ...

    async def start(self) -> None:
        """Override base class to avoid double-publish from queue loop."""
        while True:
            if self.stop_requested:
                logger.info("Stopping XPSOperator...")
                for listener in self.listeners:
                    await listener.stop()
                break
            message = await self.listener_queue.get()
            await self.process(message)
```

### 2. `labview.py`

Replace the external `setup_zmq()` function with a self-contained `__init__` that builds
the ZMQ socket from simple kwargs. This allows YAML to instantiate the listener directly.

```python
class XPSLabviewZMQListener(Listener):
    """
    Listener for LabVIEW ZMQ publisher.
    Builds its own ZMQ socket from address/port kwargs
    so it can be instantiated directly from YAML config.
    """

    def __init__(
        self,
        operator: Operator,
        zmq_pub_address: str = "tcp://localhost",
        zmq_pub_port: int = 5555,
    ):
        self.operator = operator
        self.stop_signal = False
        ctx = zmq.asyncio.Context()
        self.zmq_socket = ctx.socket(zmq.asyncio.SUB)
        self.zmq_socket.setsockopt(zmq.RCVHWM, 100000)
        self.zmq_socket.connect(f"{zmq_pub_address}:{zmq_pub_port}")
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")

    async def start(self) -> None:
        # existing loop logic unchanged
        ...

    async def stop(self) -> None:
        self.stop_signal = True
```

### 3. `timepix.py`

Same pattern as `labview.py` — move ZMQ socket creation into `__init__`.

```python
class XPSTimepixZMQListener(Listener):
    """
    Listener for Timepix ZMQ publisher.
    Builds its own ZMQ socket from address/port kwargs
    so it can be instantiated directly from YAML config.
    """

    def __init__(
        self,
        operator: Operator,
        zmq_pub_address: str = "tcp://localhost",
        zmq_pub_port: int = 5657,
    ):
        self.operator = operator
        self.stop_signal = False
        ctx = zmq.asyncio.Context()
        self.zmq_socket = ctx.socket(zmq.asyncio.SUB)
        self.zmq_socket.setsockopt(zmq.RCVHWM, 100000)
        self.zmq_socket.connect(f"{zmq_pub_address}:{zmq_pub_port}")
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")

    async def start(self) -> None:
        # existing loop logic unchanged
        ...

    async def stop(self) -> None:
        self.stop_signal = True
```

### 4. `websockets.py`

Replace the `ws_url` constructor arg (already close) — ensure no pre-built objects are
required.

```python
class XPSWSResultPublisher(Publisher):
    def __init__(self, ws_url: str = "ws://localhost:8001/xps_operator"):
        super().__init__()
        self.ws_url = ws_url
        self.websocket_server = None
        self.connected_clients = set()
        self.current_start_message = None
    # rest of class unchanged
```

---

## Pipeline YAML Files

Create a `pipelines/` directory at the repo root.

### LabVIEW pipeline

```yaml
# pipelines/labview_pipeline.yaml
blocks:
  - name: XPS LabVIEW Operator
    description: Real-time AP-XPS processor for LabVIEW data
    operator:
      class: tr_ap_xps.pipeline.xps_operator.XPSOperator
      kwargs:
        build_heatmaps: false
    listeners:
      - class: tr_ap_xps.labview.XPSLabviewZMQListener
        kwargs:
          zmq_pub_address: "tcp://localhost"
          zmq_pub_port: 5555
    publishers:
      - class: tr_ap_xps.websockets.XPSWSResultPublisher
        kwargs:
          ws_url: "ws://0.0.0.0:8001/xps_operator"
```

### Timepix pipeline

```yaml
# pipelines/timepix_pipeline.yaml
blocks:
  - name: XPS Timepix Operator
    description: Real-time AP-XPS processor for Timepix data
    operator:
      class: tr_ap_xps.pipeline.xps_operator.XPSOperator
      kwargs:
        build_heatmaps: false
    listeners:
      - class: tr_ap_xps.timepix.XPSTimepixZMQListener
        kwargs:
          zmq_pub_address: "tcp://localhost"
          zmq_pub_port: 5657
    publishers:
      - class: tr_ap_xps.websockets.XPSWSResultPublisher
        kwargs:
          ws_url: "ws://0.0.0.0:8001/xps_operator"
```

### Container pipeline (Docker — overrides addresses)

```yaml
# pipelines/timepix_container_pipeline.yaml
blocks:
  - name: XPS Timepix Operator
    description: Real-time AP-XPS processor (containerised)
    operator:
      class: tr_ap_xps.pipeline.xps_operator.XPSOperator
      kwargs:
        build_heatmaps: false
    listeners:
      - class: tr_ap_xps.timepix.XPSTimepixZMQListener
        kwargs:
          zmq_pub_address: "tcp://simulator"
          zmq_pub_port: 5657
    publishers:
      - class: tr_ap_xps.websockets.XPSWSResultPublisher
        kwargs:
          ws_url: "ws://0.0.0.0:8001/xps_operator"
```

---

## Running Pipelines

```bash
# local development
arroyo run pipelines/timepix_pipeline.yaml

# specific block from a multi-block file
arroyo run pipelines/timepix_pipeline.yaml --block "XPS Timepix Operator"

# validate config without running
arroyo validate pipelines/timepix_pipeline.yaml

# list all blocks in a file
arroyo list-blocks pipelines/timepix_pipeline.yaml

# verbose logging
arroyo run pipelines/timepix_pipeline.yaml --verbose
```

---

## Docker

Update `docker-compose.yml` processor service command:

```yaml
# docker-compose.yml
processor:
  command: arroyo run /app/pipelines/timepix_container_pipeline.yaml
  build:
    context: .
    dockerfile: Dockerfile_processor
  ...
```

Update `Dockerfile_processor` to ensure `arroyopy` CLI is available:

```dockerfile
FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade .
# arroyo CLI is installed as part of arroyopy package
CMD ["arroyo", "run", "/app/pipelines/timepix_container_pipeline.yaml"]
```

---

## The Rule

> Every operator pipeline must be defined in a YAML file under `pipelines/`.
> No `*_cli.py` files. No manual operator wiring in Python.
> All listener and publisher classes must be self-contained — constructable from
> simple scalar kwargs only, with no pre-built objects required.
