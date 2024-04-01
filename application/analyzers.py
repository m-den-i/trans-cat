import re
from dataclasses import dataclass
from typing import Iterable, Sequence, Protocol

import sqlalchemy as sa

from application.models import get_text_description, ExistingResponse
from application.redis import RedisMessage


class Detector(Protocol):
    async def try_detect(self, descriptions: Iterable[str]) -> tuple[list[list[int, tuple[str, str]]], list[int]]:
        ...


class DBCheckDetector:
    def __init__(self, engine, db_table):
        self.engine = engine
        self.db_table = db_table

    async def try_detect(self, descriptions: Iterable[str]):
        found_, missing_ = await self._check_labels(descriptions)
        # Fallback
        if missing_:
            descriptions = self._post_proc_descriptions(descriptions)
            found_, missing_ = await self._check_labels(descriptions)
        return found_, missing_

    def _post_proc_descriptions(self, descriptions: Iterable[str]) -> tuple[str]:
        result = []
        for desc in descriptions:
            if "BOLT" in desc:
                parts = desc.split(" ")
                first_part, city = "/".join(parts[:-1]), parts[-1]
                result.append(f"{first_part} {city}")
            else:
                result.append(desc)
        return tuple(result)

    async def _check_labels(self, descriptions: Iterable[str]):
        descriptions = tuple(descriptions)
        _id, _description, _category = [getattr(self.db_table.c, c) for c in ("id", "description", "category")]
        query = sa.select(_id, _description, _category).where(_description.in_(descriptions))
        with self.engine.connect() as connection:
            result = tuple(connection.execute(query))
        data = {
            res[1]: (res[0], res[2]) for res in result
        }
        found: list[list[int, tuple[str, str]]] = []
        missing: list[int] = []
        for ind, description in enumerate(descriptions):
            if (id_category := data.get(description)) is not None:
                found.append([ind, id_category])
            else:
                missing.append(ind)
        return found, missing


@dataclass
class RegexDetector:
    regex_map: dict[str, str]

    def _found_match(self, desc: str) -> str | None:
        for regex, cat in self.regex_map.items():
            if re.match(regex, desc):
                return cat
        return None

    async def try_detect(self, descriptions: Iterable[str]):
        missing = []
        found = []
        for ind, desc in enumerate(descriptions):
            if (cat := self._found_match(desc)) is not None:
                found.append(
                    [ind, ("RE", cat)]
                )
            else:
                missing.append(ind)
        return found, missing


@dataclass
class Analyzer:
    detectors: Sequence[Detector]

    async def analyze(self, msgs: Sequence[RedisMessage]):
        descriptions = {ind: get_text_description(msg) for ind, msg in enumerate(msgs)}
        found_: list[list[int, tuple[str, str]]] = []
        missing_: list[int] = [x for x in range(len(descriptions))]
        for detector in self.detectors:
            _descriptions = [descriptions[ind] for ind in missing_]
            _found, missing_ = await detector.try_detect(_descriptions)
            found_.extend(_found)
            if not missing_:
                break

        found: list[ExistingResponse] = []
        for ind, id_category in found_:
            msg = msgs[ind]
            id_, category = id_category
            response = ExistingResponse(
                key=msg.key,
                found_key=id_,
                description=descriptions[ind],
                category=category,
                amount=msg.data.amount,
            )
            found.append(response)
        missing = [msgs[ind] for ind in missing_]
        return found, missing
