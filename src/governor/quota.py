"""Hierarchical quota tree with lease semantics.

A subagent never receives budget from the global pool: it receives a slice
*carved out of its parent's remaining quota* (``spawn_child``). The invariant

    sum(live children's allocations) + local spend <= own allocation

holds at every node, so total commitment is bounded by the root allocation no
matter how deep or wide the spawn tree grows. Spawning is a reinforcing loop;
lease inheritance caps its gain (Meadows' leverage point #7): each level can
only subdivide what its parent had left, so a runaway spawn cascade decays
geometrically instead of growing exponentially. Closing a node returns its
unspent quota to the parent, so nothing leaks.

Decisions are local: an agent checks its own node without touching shared
state, which removes admission contention entirely. The price is that the
control is approximate -- system-wide overshoot is bounded by the sum of
in-flight reservations across leaves, at most (number of concurrent leaves x
worst-case call cost) -- absorbed in practice by a soft-limit margin.
"""

from __future__ import annotations

from dataclasses import dataclass


class QuotaError(Exception):
    pass


@dataclass
class LocalReservation:
    amount: int
    settled: bool = False


class QuotaNode:
    """One agent's budget lease inside the spawn tree."""

    def __init__(self, name: str, allocation: int, parent: "QuotaNode | None" = None):
        self.name = name
        self.allocation = allocation
        self.parent = parent
        self.spent = 0
        self.committed = 0
        self.leased = 0  # currently allocated to live children
        self.children: list[QuotaNode] = []
        self.closed = False

    @property
    def remaining(self) -> int:
        return self.allocation - self.spent - self.committed - self.leased

    def spawn_child(self, name: str, amount: int) -> "QuotaNode":
        """Carve `amount` out of this node's remaining quota for a subagent."""
        if self.closed:
            raise QuotaError(f"{self.name}: cannot spawn from a closed node")
        if amount > self.remaining:
            raise QuotaError(
                f"{self.name}: lease of {amount} exceeds remaining {self.remaining}"
            )
        self.leased += amount
        child = QuotaNode(name, amount, parent=self)
        self.children.append(child)
        return child

    def try_consume(self, estimate: int) -> LocalReservation | None:
        """Local admission check -- no shared state, no lock needed."""
        if self.closed or estimate > self.remaining:
            return None
        self.committed += estimate
        return LocalReservation(amount=estimate)

    def settle(self, reservation: LocalReservation, actual: int) -> None:
        if not reservation.settled:
            self.committed -= reservation.amount
            reservation.settled = True
        self.spent += actual

    def close(self) -> None:
        """Terminate this agent; unspent quota reverts to the parent."""
        if self.closed:
            return
        for child in list(self.children):  # copy: child.close() mutates this list
            child.close()
        self.closed = True
        if self.parent is not None:
            self.parent.leased -= self.allocation
            self.parent.spent += self.spent
            # the difference (allocation - spent) silently becomes available
            # again in parent.remaining
            # Drop the closed child from the parent so tree_spent() does not
            # count its spend twice (once rolled into parent.spent above, once
            # via the parent's own recursion) and a long-lived parent does not
            # accumulate dead children.
            if self in self.parent.children:
                self.parent.children.remove(self)

    def tree_spent(self) -> int:
        """Total tokens actually consumed in this subtree."""
        own = self.spent
        if not self.closed:
            own += sum(c.tree_spent() for c in self.children)
        return own

    def tree_size(self) -> int:
        return 1 + sum(c.tree_size() for c in self.children)
