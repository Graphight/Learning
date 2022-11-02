import pickle
from dataclasses import dataclass


@dataclass
class TwitterHelper:
    api_key = "YOUR_TWITTER_API_KEY"
    secret_key = "YOUR_TWITTER_SECRET_KEY"
    bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
    access_token = "YOUR_TWITTER_ACCESS_TOKEN"
    access_secret = "YOUR_TWITTER_ACCESS_SECRET"

@dataclass
class RedditHelper:
    app_id = "YOUR_REDDIT_APP_ID"
    secret = "YOUR_REDDIT_SECRET"



twitter_helper = TwitterHelper()
reddit_helper = RedditHelper()

save_helper = open(f"twitter_helper.pickle", "wb")
pickle.dump(twitter_helper, save_helper)
save_helper.close()

save_helper = open(f"reddit_helper.pickle", "wb")
pickle.dump(reddit_helper, save_helper)
save_helper.close()
