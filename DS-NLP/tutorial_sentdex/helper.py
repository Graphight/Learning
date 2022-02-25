import pickle
from dataclasses import dataclass


@dataclass
class TwitterHelper:
    api_key = "hS26fQQsvMvRNGkWOuIQdRJ7L"
    secret_key = "jRFAtBVnMb29VfHNYpkWvfDZqEYIANYYF2hCp2FYs0C0NvxNAK"
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAO0LOgEAAAAAHZWcUtqQRnqjL1ek6kQC93ojfWA%3DLnXHO4UKcsqs9Xpx3yssdDVMDoaUbkBcWeyqT5NRXL9mV2Z71W"
    access_token = "775301646058336256-rCarTByH1FVjHUd5vQF03CMlkk7COwY"
    access_secret = "sFrAox4UWxIhKpQKBmAfnQSBFjK6jkAD883j0q7Ppo78W"

@dataclass
class RedditHelper:
    app_id = "cF1HTbuhBmbBeg"
    secret = "fMpmpQE_0iO9qc-OuCg3uKksOos7Dw"



twitter_helper = TwitterHelper()
reddit_helper = RedditHelper()

save_helper = open(f"twitter_helper.pickle", "wb")
pickle.dump(twitter_helper, save_helper)
save_helper.close()

save_helper = open(f"reddit_helper.pickle", "wb")
pickle.dump(reddit_helper, save_helper)
save_helper.close()
