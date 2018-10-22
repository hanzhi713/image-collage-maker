"""
extract_img.py
get images from WeChat friends list.
"""
import itchat
import os
import pickle
from concurrent.futures import *
import concurrent.futures
import time
from tqdm import tqdm
from math import ceil


def download(args):
    user_name, image_id = args
    itchat.get_head_img(userName=user_name, picDir='img/%d.png' % image_id)
    return args


if __name__ == "__main__":
    # 登陆
    print("Logging in...")
    itchat.auto_login(hotReload=True)

    # 获取通讯录列表
    print("Loading contact...")
    friends = itchat.get_friends(update=True)

    if not os.path.isdir('img'):
        os.mkdir('img')

    if os.path.isfile("img/cache.pkl"):
        downloaded = pickle.load(open("img/cache.pkl", "rb"))
        assert type(downloaded) == dict
    else:
        downloaded = {}

    num_friends = len(friends)
    max_wait_time = 60

    while len(downloaded) < len(friends):
        available_numbers = [i for i in range(num_friends) if i not in downloaded.values()]

        if len(downloaded) > 0:
            print(available_numbers)

        pool = ThreadPoolExecutor(len(available_numbers))

        # use multi-threading to accelerate the download process
        f = []
        counter = 0
        for friend in friends:
            if not friend['UserName'] in downloaded:
                f.append(pool.submit(download, (friend['UserName'], available_numbers[counter])))
                counter += 1

        start_time = time.clock()
        for i, future in tqdm(enumerate(f), total=len(f), desc="[Downloading images]", unit="imgs"):
            try:
                if time.clock() - start_time > max_wait_time:
                    user_name, idx = future.result(0)
                else:
                    user_name, idx = future.result(ceil(max_wait_time - time.clock() + start_time))
                downloaded[user_name] = idx
            except concurrent.futures.TimeoutError:
                print("\nTimeout when downloading the head image of", friends[available_numbers[i]]['NickName'])

        if len(downloaded) < len(friends):
            print("Warning: Failed to downlodad some of the images")
            print("Retrying...")

        pickle.dump(downloaded, open("img/cache.pkl", "wb"))

    print("Success")
    exit(0)