from dataclasses import dataclass, field
from typing import List


@dataclass
class GameType:
    id: int = 0
    name: str = ""
    release: str = ""
    rawg_rating: float = 0.0
    igdb_rating: float = 0.0
    hltb_rating: float = 0.0
    metacritic_rating: float = 0.0
    user_rating: float = 0.0
    platforms: List[str] = field(default_factory=list)
    main_story: float = 0.0
    main_extra: float = 0.0
    completionist: float = 0.0
    cover_url: List[str] = field(default_factory=list)
    developers: List[str] = field(default_factory=list)
    publishers: List[str] = field(default_factory=list)
    description: str = ""
    language_supports: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class IGDBType(GameType):
    game_modes: List[str] = field(default_factory=list)
    game_type: str = ""
    player_perspectives: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)


@dataclass
class RAWGType(GameType):
    esrb_rating: str = ""


@dataclass
class HLTBType(GameType):
    game_type: str = ""


@dataclass
class GamespotType(GameType):
    themes: List[str] = field(default_factory=list)


@dataclass
class MetacriticType(GameType):
    slug: str = ""
