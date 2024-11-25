"""Utilities for interacting with the RapidAPI Twitter45 API."""
import IPython.display as ipy
import functools

from api_schemas import *
from api_utils import *


class TwitterAPI45(RapidAPI):
    def __init__(self, key: str) -> None:
        super().__init__(key, "twitter-api45")

    @RapidAPI.method("GET")
    def following(self, screenname: str, cursor: Optional[str] = None, rest_id: Optional[str] = None) -> UserFollowing:
        """
        Get the users that a user follows.
        :param screenname: The screen name of the user to fetch the following for.
        :param cursor: An optional cursor pointing to a specific page of results.
        :param rest_id: An optional rest ID pointing to a specific user. Overrides screenname, and is generally faster.
        :return: A UserFollowing object.
        """
        pass

    @RapidAPI.method("GET")
    def followers(self, screenname: str, cursor: Optional[str] = None) -> UserFollowers:
        """
        Get the users that follow a user.
        :param screenname: The screen name of the user to fetch the followers for.
        :param cursor: An optional cursor pointing to a specific page of results.
        :return: A UserFollowers object.
        """
        pass

    @RapidAPI.method("GET")
    def replies(self, screenname: str, cursor: Optional[str] = None) -> UserReplies:
        """
        Get the replies made by a user.
        :param screenname: The screen name of the user to fetch the replies for.
        :param cursor: An optional cursor pointing to a specific page of results.
        :return: A UserReplies object.
        """
        pass

    @RapidAPI.method("GET")
    def search(self, query: str, cursor: Optional[str] = None, search_type: Optional[str] = None) -> SearchResults:
        """
        Search for tweets containing a query.
        :param query: The query to search for.
        :param cursor: An optional cursor pointing to a specific page of results.
        :param search_type: An optional search type.
        :return: A SearchResults object.
        """
        pass

    @RapidAPI.method("GET")
    def screenname(self, screenname: str, rest_id: Optional[str] = None) -> UserDataShort:
        """
        Get information about a user.
        :param screenname: The screen name of the user to fetch information for.
        :param rest_id: An optional rest ID pointing to a specific user. Overrides screenname, and is generally faster.
        :return: A UserDataShort object.
        """
        pass

    @RapidAPI.method("GET")
    def timeline(self, screenname: str, rest_id: Optional[str] = None, cursor: Optional[str] = None) -> UserTimeline:
        """
        Get the timeline of a user.
        :param screenname: The screen name of the user to fetch the timeline for.
        :param rest_id: An optional rest ID pointing to a specific user. Overrides screenname, and is generally faster.
        :param cursor: An optional cursor pointing to a specific page of results.
        :return: A UserTimeline object.
        """
        pass

twitter = TwitterAPI45(rapidapi_key)
def get_followers(screen_name: str, limit: int = 1000, log_file: Optional[str] = None) -> List[str]:
    result = cursor_traverse(
        twitter.followers,
        {"screenname": screen_name},
        "next_cursor",
        "followers",
        max_fetches = limit // 50,
        log_file = log_file
    )
    return result["data"], result["cursor"]

def get_following(screen_name: str, limit: int = 1000, log_file: Optional[str] = None) -> List[str]:
    result = cursor_traverse(
        twitter.following,
        {"screenname": screen_name},
        "next_cursor",
        "following",
        max_fetches = limit // 50,
        log_file = log_file
    )
    return result["data"], result["cursor"]

def get_timeline(screen_name: str, limit: int = 10, log_file: Optional[str] = None) -> List[str]:
    result = cursor_traverse(
        twitter.timeline,
        {"screenname": screen_name},
        "next_cursor",
        "timeline",
        max_fetches = limit,
        log_file = log_file
    )
    return result["data"], result["cursor"]

def get_replies(screen_name: str, limit: int = 10, log_file: Optional[str] = None) -> List[str]:
    result = cursor_traverse(
        twitter.replies,
        {"screenname": screen_name},
        "next_cursor",
        "timeline",
        max_fetches = limit,
        log_file = log_file
    )
    return result["data"], result["cursor"]

def make_html_widget(record: Dict[str, Any], config: Dict[str, Any] = {}) -> str:
    cfg = {
        "root": None,
        "rel": {
            "if_you_follow": True,
            "if_follows_you": True,
            "then_you_follow": True,
            "then_follows_you": True
        },
        "expn": 0.5,  # in [0, 1]; higher values more strongly penalize large follow counts (no 'casting wide nets')
        "min": {"total": 20, "known": 3},
        "fn": lambda expn: {
            "accuracy": lambda follow_count: 1.0 / follow_count**expn,
            "precision": lambda num_users, proportion, follower_count: proportion * (num_users / follower_count)**expn
        },
        "html": {
            "height": "600px",
            "top_n": 40
        }
    }
    cfg["fn"] = cfg["fn"](cfg["expn"])
    for k, v in config.items():
        if isinstance(v, dict):
            cfg[k] |= v
        elif v is not None:
            cfg[k] = v
    if cfg["root"] is None:
        raise ValueError("Root user must be specified via the 'root' key in the config.")
    if_you_follow, if_follows_you = cfg["rel"]["if_you_follow"], cfg["rel"]["if_follows_you"]
    then_you_follow, then_follows_you = cfg["rel"]["then_you_follow"], cfg["rel"]["then_follows_you"]

    first_order = (record["users"][cfg["root"]]["follows_from"] if if_you_follow else []) + (record["users"][cfg["root"]]["follows_to"] if if_follows_you else [])

    sum_weights, scores = {}, []
    for alice in first_order:
        r_alice = record["users"][alice]
        follows_from = r_alice["follows_from"]
        follows_to = r_alice["follows_to"]
        friends_count = r_alice["data"]["friends_count"]
        followers_count = r_alice["data"]["followers_count"]
        if follows_from is None or follows_to is None:
            continue
        users = (follows_from if then_follows_you else []) + (follows_to if then_you_follow else [])
        count = len(users)
        if users and count != 0:
            weight = cfg["fn"]["accuracy"](count)
            for bob in users:
                sum_weights[bob] = sum_weights.get(bob, 0) + weight + 1j
                # this is a complex number to store both the combined weight of all alices and their numerosity

    for (bob, score) in sum_weights.items():
        if (r_bob := record["users"].get(bob)):
            b_sel = (r_bob["data"]["followers_count"] if then_follows_you else 0) + (r_bob["data"]["friends_count"] if then_you_follow else 0)
            if b_sel < cfg["min"]["total"] or score.imag < cfg["min"]["known"]:
                continue
            sum_weight, num_users = score.real, score.imag
            scores.append((
                (bob, r_bob["data"]["name"], r_bob["data"]["profile_image"]),
                sum_weight,
                (b_sel, num_users),
                cfg["fn"]["precision"](num_users, sum_weight, b_sel)
            ))
    rel_1 = 'know' if (then_you_follow and then_follows_you) else ('follow' if then_you_follow else 'be followed by')
    rel_2 = 'know' if (if_you_follow and if_follows_you) else ('follow' if if_you_follow else "are followed by")

    top_n = sorted(scores, key = lambda x: x[-1], reverse = True)[:cfg["html"]["top_n"]]
    root_at, root_name, root_img = top_n[0][0]

    html_code = """
    <style>.xl {font-size: 1.8em;} .lg {font-size: 1.5em;} .md {font-size: 1.25em;} .sans {font-family: 'Avenir Next';} .mono {font-family: 'Fira Code';} .row {display: flex; flex-direction: row; align-items: center;} .col {display: flex; flex-direction: column; justify-content: space-between; gap: 0.25em;} .circle {border-radius: 50%;} .id-box {height: 1.5em; overflow: clip;} .rank-box {width: 2em; margin: 0em 0.5em;}</style>""" + f"""
    <div style='margin: 0em 0em 2em 0em;'>
        <div.sans row>
            <span.xl>Who are you <em>disproportionately</em> likely to {rel_1} if you {rel_2}</span>
            <div.row style='border: 1px solid white; border-radius: 8px; padding: 0.5em; margin: 0em 1em; background-color: #1B1519;'>
                <img.circle src='{root_img}'/>
                <div.mono col style='margin-left: 0.5em; gap: 0em;'>
                    <div.md>{root_name}</div>
                    <div><a href='https://x.com/{root_at}'>@{root_at}</a></div>
                </div>
            </div>
            <span.xl>?</span>
        </div>
    </div>
    <div.mono, style='width: min-content; white-space: break-spaces; display: flex; flex-direction: column; height: {cfg["html"]["height"]}; flex-wrap: wrap;'>
    """ + "\n".join((lambda at, name, prof_pic, prec: f"""
        <div.row style='height: 75px;'>
            <div.lg rank-box>{prec / top_n[1][-1]:.2f}</div>
            <img.circle src='{prof_pic}'/>
            <div.col style='margin-left: 1.0em;'>
                <div.md id-box>{name}</div>
                <div><a href='https://x.com/{at}'>@{at}</a></div>
            </div>
        </div>
    """)(*data, score) for (data, raw, f_corr, score) in top_n[1:]) + """
    </div>
    """
    for (rx, sub) in [(r"<(\w+)\.([\w\- ]+?)([\w\-]+=|>)", r"<\1 class='\2' \3")]:
        html_code = re.sub(rx, sub, html_code)
    return html_code
