"""
Game state for the win-probability engine.

- GameState: minimal state at a moment in the game (time, score, period, possession).
- from_event_row(): build GameState from a parsed event (dict or row).
- apply_event(state, event): return state after applying one event (for replay).
- game_seconds_remaining(): total seconds left in game from (period, period_seconds_left).
"""
from dataclasses import dataclass

# NBA: 12 min regulation per period, 5 min OT
SECONDS_PER_REGULATION_PERIOD = 12 * 60  # 720
SECONDS_PER_OT_PERIOD = 5 * 60            # 300


def game_seconds_remaining(period: int, period_seconds_left: float) -> float:
    """
    Total seconds left in the game from current period and clock.

    Regulation: period 1-4, 720 sec each. OT: period 5+, 300 sec each.
    """
    if period <= 0:
        return 4 * SECONDS_PER_REGULATION_PERIOD
    if period <= 4:
        return (4 - period) * SECONDS_PER_REGULATION_PERIOD + period_seconds_left
    # OT: period 5 = 0–300 sec left, period 6 = 300–600, etc.
    return (period - 5) * SECONDS_PER_OT_PERIOD + period_seconds_left


@dataclass
class GameState:
    """
    Minimal state at a moment in the game for win-probability modeling.

    - time_remaining_sec: total seconds left in the game
    - score_diff: home score minus away score
    - period: quarter (1-4 regulation, 5+ OT)
    - possession: "home" | "away" (who has the ball)
    - is_home_court: whether the "home" team is at home (for our data, always True)
    """
    time_remaining_sec: float
    score_diff: int
    period: int
    possession: str
    is_home_court: bool = True

    @classmethod
    def from_event_row(cls, row: dict) -> "GameState":
        """
        Build a GameState from a parsed event row (e.g. one row of events_dataset or game events).

        Expects keys: period, time_remaining_sec, home_score, away_score, possession.
        """
        period = int(row.get("period", 1))
        period_sec = float(row.get("time_remaining_sec", 0))
        home = int(row.get("home_score", 0))
        away = int(row.get("away_score", 0))
        poss = row.get("possession") or ""
        if poss not in ("home", "away"):
            poss = "home"
        return cls(
            time_remaining_sec=game_seconds_remaining(period, period_sec),
            score_diff=home - away,
            period=period,
            possession=poss,
            is_home_court=True,
        )

    def to_feature_dict(self) -> dict:
        """For training: state as a flat dict (e.g. for pandas/ML)."""
        return {
            "time_remaining_sec": self.time_remaining_sec,
            "score_diff": self.score_diff,
            "period": self.period,
            "possession_home": 1 if self.possession == "home" else 0,
            "is_home_court": 1 if self.is_home_court else 0,
        }


def initial_state() -> GameState:
    """State at tip-off (period 1, 12:00, 0-0)."""
    return GameState(
        time_remaining_sec=4 * SECONDS_PER_REGULATION_PERIOD,
        score_diff=0,
        period=1,
        possession="home",  # arbitrary; jump ball determines first possession
        is_home_court=True,
    )


def apply_event(state: GameState, event: dict) -> GameState:
    """
    Return the GameState after applying one parsed event.

    The event row already encodes the resulting state (scores, time, period,
    possession after the play), so we build the new state from the event.
    Use this to replay a game: state = initial_state(); for event in events: state = apply_event(state, event).

    event: dict with period, time_remaining_sec, home_score, away_score, possession
           (e.g. one row from events_dataset or a game's _events.csv).
    """
    return GameState.from_event_row(event)
