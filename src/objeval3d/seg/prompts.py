"""The text prompts given to SAM 3, one per object: the final ones of the paper (Table F.1), shipped as `objeval3d/data/sam3_prompts.json`.

BOP object names are identifiers, not English. Left alone, "potted_meat_can" reaches SAM 3 as a token soup; the file lists the phrases
a person would actually say, which is what a concept segmenter is built to take. An object that is not listed is prompted with its BOP
name, underscores replaced by spaces. Prompt wording swings the result from 0 to 17 empty masks out of 18, so it is an experimental
variable: the prompts were settled by a bounded sweep on a few objects and are not to be tuned by feel (every prompt used by a run is
written to its record).
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path


def load_prompts(config: str | Path | None = None) -> dict[str, str]:
    if config:
        text = Path(config).read_text(encoding="utf-8")
    else:
        text = resources.files("objeval3d.data").joinpath("sam3_prompts.json").read_text(encoding="utf-8")
    return json.loads(text)["overrides"]


def prompt_for(obj_name: str, config: str | Path | None = None) -> str:
    return load_prompts(config).get(obj_name, obj_name.replace("_", " "))
