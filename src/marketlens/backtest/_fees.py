from __future__ import annotations

from abc import ABC, abstractmethod


class FeeModel(ABC):
    @abstractmethod
    def calculate(self, price: float, size: float, is_maker: bool) -> float: ...


class PolymarketFeeModel(FeeModel):
    """Polymarket fees (docs.polymarket.com/trading/fees, 2026):
    taker ``fee = shares * fee_rate * (p*(1-p))^exponent``, rounded to 5 d.p.

    Makers pay nothing and get ``rebate_rate`` of the taker fee their fill
    generated back, returned as a negative fee. Real rebates are pooled per
    market by maker volume, so this is an estimate.  Use :meth:`for_category`
    to pick the schedule from a market's category.
    """

    def __init__(self, fee_rate: float, exponent: float = 1, rebate_rate: float = 0.0) -> None:
        self._fee_rate = fee_rate
        self._exponent = exponent
        self._rebate_rate = rebate_rate

    # category -> (taker fee rate, maker rebate rate)
    _SCHEDULE = {
        "crypto": (0.07, 0.20),
        "sports": (0.05, 0.15),
        "finance": (0.04, 0.25),
        "politics": (0.04, 0.25),
        "economics": (0.05, 0.25),
        "culture": (0.05, 0.25),
        "weather": (0.05, 0.25),
        "mentions": (0.04, 0.25),
        "tech": (0.04, 0.25),
        "other": (0.05, 0.25),
        "general": (0.05, 0.25),
        "geopolitics": (0.0, 0.0),
    }

    @classmethod
    def crypto(cls) -> PolymarketFeeModel:
        """Crypto markets: fee_rate=0.07, 20% rebate. 0.0175/share at p=0.50."""
        return cls(0.07, rebate_rate=0.20)

    @classmethod
    def sports(cls) -> PolymarketFeeModel:
        """Sports markets: fee_rate=0.05, 15% rebate (0.0125/share at p=0.50)."""
        return cls(0.05, rebate_rate=0.15)

    _SPORTS_CATEGORIES = frozenset({
        "sports", "football", "basketball", "baseball", "hockey",
        "soccer", "tennis", "golf", "mma", "boxing", "cricket",
        "rugby", "nfl", "nba", "mlb", "nhl", "ncaab",
    })

    @classmethod
    def for_category(cls, category: str | None) -> FeeModel:
        """Return the fee model for a Polymarket market category (zero when unknown)."""
        if not category:
            return ZeroFeeModel()
        cat = category.lower()
        if cat in cls._SPORTS_CATEGORIES:
            cat = "sports"
        if cat not in cls._SCHEDULE or cls._SCHEDULE[cat][0] == 0.0:
            return ZeroFeeModel()
        rate, rebate = cls._SCHEDULE[cat]
        return cls(rate, rebate_rate=rebate)

    def calculate(self, price: float, size: float, is_maker: bool) -> float:
        taker = round(size * self._fee_rate * (price * (1.0 - price)) ** self._exponent, 5)
        return -round(taker * self._rebate_rate, 5) if is_maker else taker


class ZeroFeeModel(FeeModel):
    """Always returns 0."""

    def calculate(self, price: float, size: float, is_maker: bool) -> float:
        return 0.0


class FlatFeeModel(FeeModel):
    """Fixed fee per share."""

    def __init__(self, fee_per_share: float) -> None:
        self._fee = fee_per_share

    def calculate(self, price: float, size: float, is_maker: bool) -> float:
        return self._fee * size
