# from translate import Translator
#
#
# def translate_text(text: str) -> str:
#     translator = Translator(to_lang="en")
#     translation = translator.translate(text)
#     return translation
#
#
# print(translate_text("你好"))
import requests
import hashlib
import random
import string


def translate_text_baidu(text: str, appid: str, secret_key: str) -> str:
    api_url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    from_lang = "zh"
    to_lang = "en"
    salt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    sign = appid + text + salt + secret_key
    md5 = hashlib.md5()
    md5.update(sign.encode('utf-8'))
    sign = md5.hexdigest()

    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": appid,
        "salt": salt,
        "sign": sign
    }

    response = requests.get(api_url, params=params)
    response_json = response.json()

    if "trans_result" in response_json:
        return response_json["trans_result"][0]["dst"]
    else:
        return "Error: " + response_json["error_msg"]


# 使用你的appid和密钥
appid = "20230522001685687"
secret_key = "2EJ6qEDjODT_rkeK796Q"
print(translate_text_baidu("{{{masterpiece}}},精致五官,masterpiece,完美五官,黑色的头发,精致细节,眼睛,头发氛围,高分辨率,4K画质,最佳光线,细腻肌肤,细节光线,大师作品,best quality,the highest quality ,杰作,(大师杰作),极致细节,(精致五官),(精致头发刻画),(精致眼睛刻画),精致手绘,4k画质,绚丽光影,气泡水滴,破碎感, 精致发型，柔软凌乱的头发，精致的辫子包子，精致的珍珠贝壳发饰，红唇，神韵，仙气，水雾，超细节，精致旗袍，精致薄纱，半身，立绘，花朵，合一，（超- 详细CG：1.2) , (8K: 1.2), 室内, , {{{ {{wind blowing}}}}} , glowing light, {{{{{{{half body}} }}}}}}, { {{{{光粒子}}}}}，非常详细", appid, secret_key))
