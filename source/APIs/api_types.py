from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IGDBType:
    id: int
    name: str
    game_modes: List[str]
    game_type: str
    keywords: List[str]
    language_supports: List[str]
    platforms: List[str]
    player_perspectives: List[str]
    themes: List[str]
    rating: Optional[float]
    first_release_date: Optional[int]
    genres: List[str]
    cover_url: Optional[str]
    summary: Optional[str]


@dataclass
class RAWGType:
    id: int
    name: str
    released: Optional[str]
    rating: Optional[float]
    metacritic: Optional[int]
    playtime: Optional[int]
    platforms: List[str]
    genres: List[str]
    tags: List[str]
    esrb_rating: Optional[str]
    developers: List[str]
    publishers: List[str]
    description: Optional[str]


@dataclass
class HLTBType:
    game_name: str
    game_type: str
    review_score: Optional[float]
    profile_platforms: List[str]
    release_world: Optional[str]
    main_story: Optional[float]
    main_extra: Optional[float]
    completionist: Optional[float]


@dataclass
class GamespotType:
    id: int
    name: str
    themes: List[str]
    release_date: Optional[int]
    genres: List[str]
    cover_url: Optional[str]
    description: Optional[str]


@dataclass
class MetacriticType:
    name: str
    release_date: Optional[str]
    developers: List[str]
    publishers: List[str]
    genres: List[str]
    platforms: List[str]
    critic_score: Optional[int]
    user_score: Optional[int]


@dataclass
class GameType:
    id: int
    name: str
    released: str
    rawg_rating: Optional[float]
    igdb_rating: Optional[float]
    hltb_rating: Optional[float]
    metacritic_rating: Optional[float]
    user_rating: Optional[float]
    platforms: List[str]
    main_story: Optional[float]
    main_extra: Optional[float]
    completionist: Optional[float]
    cover_url: List[str]
    developers: List[str]
    publishers: List[str]
    description: Optional[str]
    language_supports: List[str]
    genres: List[str]
    keywords: List[str]
