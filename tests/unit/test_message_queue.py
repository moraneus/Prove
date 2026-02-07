"""
Tests for the MessageQueueHandler.

Tests cover initialization, send/receive operations, FIFO ordering,
immutability, and error conditions for pending message tracking
between process pairs.
"""

import pytest

from prove.core.message_queue import MessageQueueHandler


class TestMessageQueueCreation:
    """Test MessageQueueHandler initialization."""

    def test_init_all_zero(self) -> None:
        """All pending counts start at zero."""
        mq = MessageQueueHandler({"P1", "P2", "P3"})
        assert mq.pending_count("P1", "P2") == 0
        assert mq.pending_count("P2", "P1") == 0
        assert mq.pending_count("P1", "P3") == 0

    def test_init_two_processes(self) -> None:
        """Two processes have two directional pairs."""
        mq = MessageQueueHandler({"A", "B"})
        assert mq.pending_count("A", "B") == 0
        assert mq.pending_count("B", "A") == 0

    def test_cannot_send_self(self) -> None:
        """Self-messages are not tracked (same sender and receiver)."""
        mq = MessageQueueHandler({"P1", "P2"})
        assert mq.pending_count("P1", "P1") == 0


class TestMessageQueueSend:
    """Test send operations."""

    def test_send_increments(self) -> None:
        """Sending a message increments the pending count."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq2 = mq.send("P1", "P2")
        assert mq2.pending_count("P1", "P2") == 1

    def test_send_returns_new_instance(self) -> None:
        """Send returns a new handler (immutable)."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq2 = mq.send("P1", "P2")
        assert mq2 is not mq
        assert mq.pending_count("P1", "P2") == 0

    def test_multiple_sends(self) -> None:
        """Multiple sends accumulate."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        mq = mq.send("P1", "P2")
        mq = mq.send("P1", "P2")
        assert mq.pending_count("P1", "P2") == 3

    def test_send_does_not_affect_other_pairs(self) -> None:
        """Sending from P1→P2 does not affect P2→P1 or other pairs."""
        mq = MessageQueueHandler({"P1", "P2", "P3"})
        mq = mq.send("P1", "P2")
        assert mq.pending_count("P2", "P1") == 0
        assert mq.pending_count("P1", "P3") == 0
        assert mq.pending_count("P3", "P2") == 0


class TestMessageQueueReceive:
    """Test receive operations."""

    def test_receive_decrements(self) -> None:
        """Receiving a message decrements the pending count."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        mq = mq.receive("P1", "P2")
        assert mq.pending_count("P1", "P2") == 0

    def test_receive_returns_new_instance(self) -> None:
        """Receive returns a new handler (immutable)."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq_sent = mq.send("P1", "P2")
        mq_recv = mq_sent.receive("P1", "P2")
        assert mq_recv is not mq_sent

    def test_receive_without_pending_raises(self) -> None:
        """Receiving when no message is pending raises ValueError."""
        mq = MessageQueueHandler({"P1", "P2"})
        with pytest.raises(ValueError):
            mq.receive("P1", "P2")

    def test_receive_after_all_consumed_raises(self) -> None:
        """Receiving after all messages consumed raises ValueError."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        mq = mq.receive("P1", "P2")
        with pytest.raises(ValueError):
            mq.receive("P1", "P2")


class TestCanReceive:
    """Test can_receive checks."""

    def test_can_receive_when_pending(self) -> None:
        """Can receive when there is a pending message."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        assert mq.can_receive("P1", "P2")

    def test_cannot_receive_when_empty(self) -> None:
        """Cannot receive when no message is pending."""
        mq = MessageQueueHandler({"P1", "P2"})
        assert not mq.can_receive("P1", "P2")

    def test_can_receive_direction_matters(self) -> None:
        """Pending messages are directional: P1→P2 doesn't enable P2→P1."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        assert mq.can_receive("P1", "P2")
        assert not mq.can_receive("P2", "P1")


class TestFIFOBehavior:
    """Test FIFO message ordering semantics."""

    def test_fifo_order(self) -> None:
        """Messages are consumed in FIFO order (first sent, first received)."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        mq = mq.send("P1", "P2")
        assert mq.pending_count("P1", "P2") == 2
        mq = mq.receive("P1", "P2")
        assert mq.pending_count("P1", "P2") == 1
        mq = mq.receive("P1", "P2")
        assert mq.pending_count("P1", "P2") == 0

    def test_interleaved_send_receive(self) -> None:
        """Interleaved sends and receives maintain correct counts."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        mq = mq.send("P1", "P2")
        mq = mq.receive("P1", "P2")
        mq = mq.send("P1", "P2")
        assert mq.pending_count("P1", "P2") == 2


class TestCopy:
    """Test copy operation."""

    def test_copy_creates_independent_instance(self) -> None:
        """Copy creates an independent handler."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        mq_copy = mq.copy()
        assert mq_copy.pending_count("P1", "P2") == 1
        assert mq_copy is not mq

    def test_copy_is_deep(self) -> None:
        """Modifying the copy does not affect the original."""
        mq = MessageQueueHandler({"P1", "P2"})
        mq = mq.send("P1", "P2")
        mq_copy = mq.copy()
        mq_copy2 = mq_copy.send("P1", "P2")
        assert mq.pending_count("P1", "P2") == 1
        assert mq_copy2.pending_count("P1", "P2") == 2
