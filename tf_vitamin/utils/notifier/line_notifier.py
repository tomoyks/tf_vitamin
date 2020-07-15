import requests


class LineNotify:
    """
    send notification via line notify api.
        api page: https://notify-bot.line.me/ja/
    """

    def __init__(self, token):
        """
        Args:
            token: access token for authentication.
        """
        self.token = token
        self.endpoint_url = "https://notify-api.line.me/api/notify"

    def post_msg(self, msg):
        """
        simply send single message.
        """
        post_params = self.init_post_params()
        post_params["data"] = {"message": msg}
        _ = self.post(post_params)

    def post_img(self, img):
        """
        simply send single image.
        """
        post_params = self.init_post_params()
        post_params['files'] = {"imageFile": open("figure.png", "rb")}
        _ = self.post(post_params)

    def post(self, post_params):
        r = requests.post(self.endpoint_url, **post_params)
        # TODO: add warnings when post fails.
        # if r.status_code != 200:
        #     print(f"failed to send msg {msg}")

        return r

    def init_post_params(self):
        post_params = {
            'headers': {"Authorization": "Bearer " + self.token},
        }

        return post_params