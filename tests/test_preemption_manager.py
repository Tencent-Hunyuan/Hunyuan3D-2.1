"""Unit tests for PreemptionManager."""

import threading

import pytest

from api import PreemptedError, PreemptionManager


class TestBegin:
    def test_creates_fresh_cancel_event(self, fresh_preemption_manager):
        pm = fresh_preemption_manager
        cancel = pm.begin()
        try:
            assert isinstance(cancel, threading.Event)
            assert not cancel.is_set()
        finally:
            pm.end()

    def test_cancel_previous_sets_old_event(self, fresh_preemption_manager):
        pm = fresh_preemption_manager
        cancel1 = pm.begin()
        try:
            # Second begin in another thread to avoid deadlock
            ready = threading.Event()
            cancel2_holder = [None]

            def second():
                cancel2_holder[0] = pm.begin(cancel_previous=True)
                ready.set()

            t = threading.Thread(target=second)
            t.start()
            # end() releases the lock so the second begin can proceed
            pm.end()
            ready.wait(timeout=5)
            t.join(timeout=5)

            assert cancel1.is_set(), "First cancel event should be set by second begin"
            assert not cancel2_holder[0].is_set()
        finally:
            pm.end()  # release second begin's slot

    def test_cancel_previous_false_preserves_old_event(self, fresh_preemption_manager):
        pm = fresh_preemption_manager
        cancel1 = pm.begin()
        try:
            ready = threading.Event()
            cancel2_holder = [None]

            def second():
                cancel2_holder[0] = pm.begin(cancel_previous=False)
                ready.set()

            t = threading.Thread(target=second)
            t.start()
            pm.end()
            ready.wait(timeout=5)
            t.join(timeout=5)

            assert not cancel1.is_set(), "First event should NOT be set (cancel_previous=False)"
            assert not cancel2_holder[0].is_set()
        finally:
            pm.end()


class TestEnd:
    def test_releases_processing_lock(self, fresh_preemption_manager):
        pm = fresh_preemption_manager
        pm.begin()
        pm.end()
        # Should be able to begin again immediately
        cancel = pm.begin()
        assert not cancel.is_set()
        pm.end()


class TestCancel:
    def test_cancel_when_processing(self, fresh_preemption_manager):
        pm = fresh_preemption_manager
        cancel = pm.begin()
        try:
            result = pm.cancel()
            assert result is True
            assert cancel.is_set()
        finally:
            pm.end()

    def test_cancel_when_idle(self, fresh_preemption_manager):
        pm = fresh_preemption_manager
        result = pm.cancel()
        assert result is False


class TestCheck:
    def test_raises_on_set_event(self):
        event = threading.Event()
        event.set()
        with pytest.raises(PreemptedError):
            PreemptionManager.check(event)

    def test_passes_on_unset_event(self):
        event = threading.Event()
        PreemptionManager.check(event)  # should not raise


class TestConcurrency:
    def test_concurrent_begin_blocks(self, fresh_preemption_manager):
        pm = fresh_preemption_manager
        cancel1 = pm.begin()

        second_entered = threading.Event()
        second_acquired = threading.Event()

        def second():
            # Signal that we're about to call begin (will block on _processing)
            second_entered.set()
            pm.begin(cancel_previous=True)
            second_acquired.set()

        t = threading.Thread(target=second)
        t.start()

        # Wait for the second thread to reach begin()
        second_entered.wait(timeout=5)
        # Give the thread a moment to actually block on acquire
        import time
        time.sleep(0.05)

        assert not second_acquired.is_set(), "Second begin() should block while first holds slot"

        pm.end()  # release the first slot
        second_acquired.wait(timeout=5)
        assert second_acquired.is_set(), "Second begin() should proceed after end()"

        pm.end()  # release the second slot
        t.join(timeout=5)
