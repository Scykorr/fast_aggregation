"""Реестр узлов: UDP-объявления, ручное добавление и опрос HTTP."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from fast_aggregation.constants import PEER_TTL_SEC
from fast_aggregation.models import Announce, ClusterNode, MachineResources, NodeLimits


@dataclass
class TrackedNode:
    node_id: str
    name: str
    host: str
    http_port: int
    rpc_port: int
    os_name: str = ""
    arch: str = ""
    preferred_backend: str = "auto"
    source: str = "discovery"
    last_seen: float = 0.0
    resources: MachineResources | None = None
    rpc_running: bool = False
    rpc_backend: str | None = None
    rpc_error: str | None = None
    last_error: str | None = None
    manual: bool = False
    rtt_ms: float | None = None
    bandwidth_mbps: float | None = None
    limits: NodeLimits | None = None

    def online(self, now: float, ttl: float) -> bool:
        return (now - self.last_seen) <= ttl

    def to_cluster_node(self, now: float, ttl: float) -> ClusterNode:
        return ClusterNode(
            node_id=self.node_id,
            name=self.name,
            host=self.host,
            http_port=self.http_port,
            rpc_port=self.rpc_port,
            online=self.online(now, ttl),
            resources=self.resources,
            preferred_backend=self.preferred_backend,
            source="manual" if self.manual else self.source,
            rtt_ms=self.rtt_ms,
            bandwidth_mbps=self.bandwidth_mbps,
            limits=self.limits,
        )

    def to_public(self, now: float, ttl: float) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "host": self.host,
            "http_port": self.http_port,
            "rpc_port": self.rpc_port,
            "os_name": self.os_name,
            "arch": self.arch,
            "preferred_backend": self.preferred_backend,
            "source": "manual" if self.manual else self.source,
            "online": self.online(now, ttl),
            "last_seen": self.last_seen,
            "age_sec": round(max(0.0, now - self.last_seen), 1),
            "resources": self.resources.to_dict() if self.resources else None,
            "rpc": {
                "running": self.rpc_running,
                "backend": self.rpc_backend,
                "error": self.rpc_error,
            },
            "last_error": self.last_error,
            "rtt_ms": self.rtt_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "limits": self.limits.to_dict() if self.limits else None,
        }


@dataclass
class NodeRegistry:
    ttl: float = PEER_TTL_SEC
    _nodes: dict[str, TrackedNode] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def upsert_announce(self, announce: Announce, now: float | None = None) -> TrackedNode:
        moment = time.time() if now is None else now
        with self._lock:
            current = self._nodes.get(announce.node_id)
            if current is None:
                current = TrackedNode(
                    node_id=announce.node_id,
                    name=announce.name,
                    host=announce.advertise_host,
                    http_port=announce.http_port,
                    rpc_port=announce.rpc_port,
                )
                self._nodes[announce.node_id] = current
            current.name = announce.name
            current.host = announce.advertise_host
            current.http_port = announce.http_port
            current.rpc_port = announce.rpc_port
            current.os_name = announce.os_name
            current.arch = announce.arch
            current.preferred_backend = announce.preferred_backend or "auto"
            current.last_seen = moment
            current.source = "manual" if current.manual else "discovery"
            return current

    def upsert_details(
        self,
        node_id: str,
        *,
        name: str | None = None,
        host: str | None = None,
        http_port: int | None = None,
        rpc_port: int | None = None,
        resources: MachineResources | None = None,
        preferred_backend: str | None = None,
        rpc_running: bool | None = None,
        rpc_backend: str | None = None,
        rpc_error: str | None = None,
        last_error: str | None = None,
        manual: bool | None = None,
        rtt_ms: float | None = None,
        bandwidth_mbps: float | None = None,
        limits: NodeLimits | None = None,
        seen: bool = True,
        now: float | None = None,
    ) -> TrackedNode:
        moment = time.time() if now is None else now
        with self._lock:
            current = self._nodes.get(node_id)
            if current is None:
                current = TrackedNode(
                    node_id=node_id,
                    name=name or node_id,
                    host=host or "",
                    http_port=http_port or 0,
                    rpc_port=rpc_port or 0,
                )
                self._nodes[node_id] = current
            if name:
                current.name = name
            if host:
                current.host = host
            if http_port:
                current.http_port = http_port
            if rpc_port:
                current.rpc_port = rpc_port
            if resources is not None:
                current.resources = resources
                current.os_name = resources.os_name
                current.arch = resources.arch
            if preferred_backend:
                current.preferred_backend = preferred_backend
            if rpc_running is not None:
                current.rpc_running = rpc_running
            if rpc_backend is not None:
                current.rpc_backend = rpc_backend
            if rpc_error is not None:
                current.rpc_error = rpc_error
            if last_error is not None:
                current.last_error = last_error
            if manual is not None:
                current.manual = manual
            if rtt_ms is not None:
                current.rtt_ms = rtt_ms
            if bandwidth_mbps is not None:
                current.bandwidth_mbps = bandwidth_mbps
            if limits is not None:
                current.limits = limits
            if seen:
                current.last_seen = moment
            return current

    def mark_error(self, node_id: str, message: str) -> None:
        with self._lock:
            current = self._nodes.get(node_id)
            if current is not None:
                current.last_error = message

    def remove(self, node_id: str) -> bool:
        with self._lock:
            return self._nodes.pop(node_id, None) is not None

    def get(self, node_id: str) -> TrackedNode | None:
        with self._lock:
            return self._nodes.get(node_id)

    def cluster_nodes(self, now: float | None = None) -> list[ClusterNode]:
        moment = time.time() if now is None else now
        with self._lock:
            nodes = [node.to_cluster_node(moment, self.ttl) for node in self._nodes.values()]
        nodes.sort(key=lambda item: (item.name.lower(), item.node_id))
        return nodes

    def public_nodes(self, now: float | None = None) -> list[dict]:
        moment = time.time() if now is None else now
        with self._lock:
            rows = [node.to_public(moment, self.ttl) for node in self._nodes.values()]
        rows.sort(key=lambda item: (str(item["name"]).lower(), str(item["node_id"])))
        return rows
