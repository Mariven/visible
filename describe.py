"""Utilities for describing Twitter users."""

import math
from api_utils import *
from openai import OpenAI
from tiktoken import get_encoding

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    try:
        with open("secrets.json") as file:
            content = json.load(file)
            openai_key = content["openai-api-key"]
    except Exception as e:
        raise Exception("No OpenAI API key found.") from e

cl100k_base = get_encoding("cl100k_base")
token_count = lambda s: len(cl100k_base.encode(s))

client = OpenAI(api_key = openai_key, base_url = "https://api.openai.com/v1")

max_workers = 25
# tier 3 rate limits for gpt-4o-mini: 5000 rpm, 4000000 tpm
openai_executor = RateLimitedExecutor(max_workers, 5000, 4000000, token_count)

record: Dict[str, Any] = {
    "users": {
        "example_name": {
            "data": {
                "user_id": "12312124123",
                "screen_name": "example_name",
                "description": "Example Description",
                "profile_image": "https://example.com/image.png",
                "statuses_count": 1000,
                "followers_count": 4,
                "friends_count": 2,
                "media_count": 1000,
                "name": "John Example"
            },
            "depth": 1,
            "follows_to": ["celebrity", "politician"],
            "follows_from": ["spambot", "spam_bot", "spam_bot_2", "spamb0t"],
            "cursors": {
                "follows_to": "asdf9j4awfck",
                "follows_from": "fj804waf0404af"
            }
        }
    }
}
messages: Dict[str, Any] = {
    "example_name": {
        "tweets": ["tweet 1", "tweet 2"],
        "replies": ["reply 1", "reply 2"],
        "cursors": {
            "tweets": "fa04wjfmcawasef",
            "replies": "faw0m0v9jg5gjf4mw"
        }
    }
}

del record["users"]["example_name"]
del messages["example_name"]

descriptions: Dict[str, Any] = {}

class Aptitude(BaseModel):
    field: str
    level: int
class Description(BaseModel):
    confidence: int
    description: str
    interests: Optional[List[str]]
    aptitudes: Optional[List[Aptitude]]

def make_message(name: str, first_n: int = 100) -> str:
    name_tweets = [x for x in messages[name]["tweets"] if x][:first_n]
    name_replies = [x for x in messages[name]["replies"] if x and x not in name_tweets][:first_n]

    # clean up tweets and replies
    name_tweets = [re.sub("https://t.co/[a-zA-Z0-9]+", "(link)", x) for x in name_tweets]
    name_replies = [re.sub("https://t.co/[a-zA-Z0-9]+", "(link)", x) for x in name_replies]

    # remove ellipses and lone links
    f_fn = lambda x: not x.endswith("\u2026") and x != "(link)"
    name_tweets = [x for x in name_tweets if f_fn(x)]
    name_replies = [x for x in name_replies if f_fn(x)]

    tweet_string = "\n\t\t".join([f"{i + 1}: {json.dumps(x)}" for i, x in enumerate(name_tweets)])
    reply_string = "\n\t\t".join([f"{i + 1}: {json.dumps(x)}" for i, x in enumerate(name_replies)])

    user_line = f"User: {record['users'][name]['data']['name']} (@{name})\n"
    desc_line = f"\tDescription: {json.dumps(record['users'][name]['data']['description'])}\n" if record['users'][name]['data']['description'] else ""

    return f"{user_line}{desc_line}\tTweets:\n\t\t{tweet_string}\n\tReplies:\n\t\t{reply_string}"

def get_description(name: str, cli: OpenAI) -> Optional[Description]:
    sys_prompt = """
    You are an expert at analyzing people's profiles and descriptions, and extracting their aptitudes and interests. Today, you will be given a Twitter user's profile and their tweets and replies. Your job is to analyze the profile and the content of the tweets and replies, and extract the user's aptitudes and interests. Your description should be a concise and to the point summary of the user's profile, personality, and interests, and should not exceed 100 words. Aptitudes are fields of expertise, and should be rated according to inferred skill on a scale of 1 to 5, with 1 indicating basic knowledge and 5 indicating expert-level proficiency. Remember that just because someone talks about a field a lot doesn't mean they're good at it; you should also consider the quality, insight, and skill evinced by the user's tweets and replies. You should generally aim for three to five aptitudes, providing at most nine. If you find that the user has no discernible aptitudes or interests, you should not include any.

    You will output your analysis in JSON format, using the Description schema. This schema allows for optional fields, so feel free to omit topics and aptitudes if you don't find any; it also allows for a confidence score, which you should set to between 1 and 5 based on how clearly the provided data supports your assessment. If there is not enough information to make a determination, for example if the user has not posted any tweets or replies, you should set the confidence to 0. In this case, you should also leave the other fields blank.

    Here is the user's profile and their tweets and replies:
    """
    out = cli.beta.chat.completions.parse(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": make_message(name)}
        ],
        response_format = Description
    )
    try:
        content = out.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error getting description for {name}: {e}. Output: {out}")
        return None

if __name__ == "__main__":
    with open("record.json", "r") as file:
        record["users"] |= json.load(file).get("users", {})

    with open("messages.json", "r") as file:
        messages |= json.load(file)

    # who is the root user? must be a string
    root = None

    to_do = []

    norm_a = math.sqrt  # normalization for alice's number of follows
    norm_b = math.sqrt  # normalization for bob's number of followers

    scores = {name: [
        0,  # score
        0,  # followers count
        []  # connections, with weights
    ] for name in record["users"]}

    follows_to = record["users"][root]["follows_to"]
    follows_from = record["users"][root]["follows_from"]
    # this logic is messed up
    for alice in follows_to:
        if (conn := record["users"][alice].get("follows_to")):
            c = len(conn)
            w = 1 / norm_a(c)
            if alice not in follows_from:
                w *= 0.1
            for bob in conn:
                if bob and bob not in follows_to:
                    scores[bob][0] += w
                    scores[bob][2].append((alice, w))

    for bob, v in scores.items():
        if v[0] > 0 and (foll := record["users"][bob].get("data", {}).get("followers_count", 0)) > 5:
            scores[bob][0] = v[0] / norm_b(foll)
            scores[bob][1] = foll
        else:
            scores[bob][0] = 0

    sorted_scores = sorted(scores.items(), key = lambda x: x[1][0], reverse = True)

    for name_data in sorted_scores:
        name = name_data[0]
        if len(make_message(name)) <= 1000:
            descriptions[name] = {"confidence": 0, "description": "", "interests": [], "aptitudes": []}
        elif descriptions[name] is None:
            to_do.append(name)

    start_time = time.time()

    def get_description_wrapper(name: str) -> Optional[Description]:
        print(f"({time.time() - start_time:.2f} seconds) {name} get_description started.")
        try:
            out = get_description(name, client)
            msg = f"({time.time() - start_time:.2f} seconds) {name} get_description done ({len(json.dumps(out))} bytes)"
        except Exception as e:
            msg = f"({time.time() - start_time:.2f} seconds) {name} get_description failed: {e}"
            out = None
        print(msg)
        return out

    results = openai_executor.map(get_description_wrapper, to_do)
    for name, desc in zip(to_do, results):
        descriptions[name] = desc

    i = 0
    # write to first absent descriptions{i}.json
    while True:
        try:
            with open(f"descriptions{i or ''}.json", "r") as file:
                i += 1
        except Exception as e:  # descriptions{i} doesn't exist
            with open(f"descriptions{i or ''}.json", "w") as file:
                json.dump(descriptions, file)
