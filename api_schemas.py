"""This file contains the API schemas for RapidAPI's Twitter45 API. Or, a rough approximation of them."""

from __future__ import annotations
from typing import *
from pydantic import BaseModel

class EntityUserMention(BaseModel):
    id_str: str
    name: str
    screen_name: str

class Entities(BaseModel):
    user_mentions: List[EntityUserMention]
    urls: List[Any]
    hashtags: List[Any]
    symbols: List[Any]
    timestamps: List[Any]

class Author(BaseModel):
    rest_id: str
    name: str
    screen_name: str
    followers_count: int
    favourites_count: int
    avatar: str
    blue_verified: bool

class VideoVariant(BaseModel):
    bitrate: int
    content_type: str
    url: str

class VideoOriginalInfo(BaseModel):
    bitrate: int
    content_type: str
    url: str

class PhotoSize(BaseModel):
    h: int
    w: int

class PhotoData(BaseModel):
    media_url_https: str
    id: str
    sizes: List[PhotoSize]

class MediaItem(BaseModel):
    id: str
    url: str
    photo: List[PhotoData]
    video: Any

class AffiliateLabel(BaseModel):
    url: str
    urlType: str

class AffiliateBadge(BaseModel):
    url: str

class UserDataShort(BaseModel):
    profile: str
    rest_id: str
    avatar: str
    desc: str
    name: str
    friends: int
    sub_count: int
    id: str

class UserData(BaseModel):
    user_id: str
    screen_name: str
    description: str
    profile_image: str
    statuses_count: int
    followers_count: int
    friends_count: int
    media_count: int
    name: str

class UserDataLong(BaseModel):
    status: str
    profile: str
    rest_id: str
    blue_verified: bool
    affiliates: Any
    business_account: List[Any]
    avatar: str
    header_image: str
    desc: str
    name: str
    website: str
    protected: Optional[bool]
    location: str
    friends: int
    sub_count: int
    statuses_count: int
    media_count: int
    created_at: str
    pinned_tweet_ids_str: List[str]
    id: str

class UserFollowing(BaseModel):
    following: List[UserData]
    next_cursor: str
    status: str
    more_users: bool


class UserFollowers(BaseModel):
    followers: List[UserData]
    next_cursor: str
    status: str

class TweetInfo(BaseModel):
    likes: int
    created_at: str
    text: str
    retweets: int
    bookmarks: int
    quotes: int
    replies: int
    lang: str
    conversation_id: str
    author: UserData
    media: List[MediaItem] | Any
    id: str

class Retweeted(BaseModel):
    id: str

class TimelineEntry0(BaseModel):
    tweet_id: str
    bookmarks: int
    created_at: str
    favorites: int
    text: str
    lang: str
    in_reply_to_status_id_str: Optional[str]
    views: str
    quotes: int
    replies: int
    retweets: int
    conversation_id: str
    media: List[MediaItem] | Any
    entities: Entities
    author: Author
    quoted: Any
    retweeted: Retweeted
    retweeted_tweet: Any

class TimelineEntry(BaseModel):
    tweet_id: str
    bookmarks: int
    created_at: str
    favorites: int
    text: str
    lang: str
    in_reply_to_status_id_str: Optional[str]
    views: str
    quotes: int
    replies: int
    retweets: int
    conversation_id: str
    media: List[MediaItem] | Any
    entities: Entities
    author: Author
    quoted: TimelineEntry0 | Any
    retweeted: Retweeted
    retweeted_tweet: TimelineEntry0 | Any

class UserTimeline(BaseModel):
    pinned: TimelineEntry
    timeline: List[TimelineEntry]
    next_cursor: str
    prev_cursor: str
    status: str

class UserReplies(BaseModel):
    timeline: List[TimelineEntry]
    next_cursor: str
    user: UserDataLong

class SearchResults(BaseModel):
    timeline: List[TimelineEntry]
    next_cursor: str
