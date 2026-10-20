"""A table as the paper prints it: strings in rows and columns, in the paper's order and precision."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Table:
    key: str  # "table3", "tableG2", ... the printed number of the paper
    title: str
    header: list[str]
    rows: list[list[str]]
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        width = len(self.header)
        out = [f"**{self.key}. {self.title}**", "", "| " + " | ".join(self.header) + " |", "|" + "---|" * width]
        out += ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in self.rows]
        out += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "header": self.header, "rows": self.rows, "notes": self.notes}


def fmt(x: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-0.00", "0.00") if set(s) <= set("-0.") else s


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def sci(p: float) -> str:
    """4.0e-4 -> '4.0 × 10^-4', the way the p-values of Table 8 are printed."""
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant} × 10^{int(exp)}"
