"""
ontick_rule.py -- swappable reversion fade RULE (config/predicate, not engine code).

CLEAN ROOM. Consumes engine WindowState + current mid, returns an Action.

Both LONG and SHORT enabled.

Triggers:
  - "close_beyond": fire as soon as mid is beyond mean +/- K*std.
        SHORT when mid > upper band (fade the up-extension).
        LONG  when mid < lower band (fade the down-extension).
  - "reenter": arm when mid goes beyond the band, fire when mid returns to TOUCH
        the band from outside (classic re-entry fade). Mirror convergence.
        SHORT when armed-above and mid <= upper band.
        LONG  when armed-below and mid >= lower band.

Debounce: one fire per excursion. After a fire the arm latch must clear (mid must
return inside well past the band, or a position must close) before re-arming. We
clear the arm only after the price crosses back to the mean side, so a single
excursion produces exactly one entry.

Stop: REAL SYMMETRIC explicit stop of `stop_pips` from the *fill* entry price.
  LONG  stop = entry - stop_pips*pip   (below)
  SHORT stop = entry + stop_pips*pip   (above)
This can never coincide with or sit past entry (the old protrusion-peak bug).

Target: the opposite band at entry time (snapshot), in mid terms.
Time-cap: tcap_sec from entry.

Entry-validity gate: assert the fill is NOT already past the stop. With a fixed
symmetric stop this is structurally impossible, but we assert it loudly.
"""
from dataclasses import dataclass


ENTER_LONG = "ENTER_LONG"
ENTER_SHORT = "ENTER_SHORT"
EXIT = "EXIT"


@dataclass
class Action:
    kind: str           # ENTER_LONG | ENTER_SHORT | EXIT | None
    reason: str = ""


class ReversionRule:
    def __init__(self, K, stop_pips, tcap_sec, trigger, pip):
        assert trigger in ("close_beyond", "reenter")
        assert stop_pips > 0, "stop must be a real positive distance"
        self.K = float(K)
        self.stop_pips = float(stop_pips)
        self.tcap_ms = int(round(tcap_sec * 1000))
        self.trigger = trigger
        self.pip = float(pip)

        # excursion arm latch: 0 none, +1 armed above (for SHORT), -1 armed below (for LONG)
        self._armed = 0

        # open position state
        self.pos = None  # dict(side, entry, stop, target, t_entry_ms)

    # ---- position helpers ----
    def in_position(self):
        return self.pos is not None

    def _open(self, side, fill_mid, target_mid, t_ms):
        if side == "LONG":
            stop = fill_mid - self.stop_pips * self.pip
            # entry-validity gate
            assert stop < fill_mid, f"LONG stop {stop} not below entry {fill_mid}"
        else:
            stop = fill_mid + self.stop_pips * self.pip
            assert stop > fill_mid, f"SHORT stop {stop} not above entry {fill_mid}"
        self.pos = dict(side=side, entry=fill_mid, stop=stop,
                        target=target_mid, t_entry_ms=t_ms)

    def _close(self):
        self.pos = None
        self._armed = 0  # require fresh excursion after a trade

    # ---- main step ----
    def on_tick(self, st, mid, t_ms):
        """
        st: WindowState. mid: current mid. Returns Action or None.
        The backtest is responsible for applying spread to convert mid->fill.
        This rule operates purely in MID space for signals (R3).
        """
        up, lo = st.band(self.K)

        # ------- manage open position (exits checked every tick) -------
        if self.pos is not None:
            p = self.pos
            if p["side"] == "LONG":
                if mid <= p["stop"]:
                    return Action(EXIT, "stop")
                if mid >= p["target"]:
                    return Action(EXIT, "target")
            else:  # SHORT
                if mid >= p["stop"]:
                    return Action(EXIT, "stop")
                if mid <= p["target"]:
                    return Action(EXIT, "target")
            if t_ms - p["t_entry_ms"] >= self.tcap_ms:
                return Action(EXIT, "tcap")
            return None

        # ------- no position: look for entry -------
        if st.std <= 0 or st.n < 5:
            return None

        if self.trigger == "close_beyond":
            if mid > up and self._armed != 1:
                self._armed = 1
                return Action(ENTER_SHORT, "close_above")
            if mid < lo and self._armed != -1:
                self._armed = -1
                return Action(ENTER_LONG, "close_below")
            # clear arm once back inside
            if lo <= mid <= up:
                self._armed = 0
            return None

        # trigger == "reenter": arm on excursion, fire on return to band
        if mid > up:
            self._armed = 1       # extended above -> arm a SHORT
        elif mid < lo:
            self._armed = -1      # extended below -> arm a LONG

        if self._armed == 1 and mid <= up:
            # returned to touch upper band from above -> fade short
            self._armed = 0
            return Action(ENTER_SHORT, "reenter_from_above")
        if self._armed == -1 and mid >= lo:
            self._armed = 0
            return Action(ENTER_LONG, "reenter_from_below")
        return None

    def target_for(self, side, st):
        """
        Reversion target = the band on the MEAN side of the entry.
        A LONG fades a DOWN extension (price fell below lo) and expects reversion
        UP toward the mean -> target = upper band `up`.
        A SHORT fades an UP extension (price rose above up) and expects reversion
        DOWN toward the mean -> target = lower band `lo`.
        (The earlier lo-for-LONG / up-for-SHORT mapping was inverted: it placed
        the target at the entry side, so it booked instantly for ~spread -- an
        artifact, caught by the WR~0 / meat<0 self-check.)
        """
        up, lo = st.band(self.K)
        return up if side == "LONG" else lo
