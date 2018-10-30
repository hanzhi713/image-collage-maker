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
import argparse


def download_friend(args):
    user_name, image_id, download_dir = args
    itchat.get_head_img(userName=user_name, picDir=os.path.join(download_dir, '%d.png' % image_id))
    return args


def download_chatroom_member(args):
    global chatroom
    user_name, image_id, download_dir = args
    itchat.get_head_img(userName=user_name, chatroomUserName=chatroom['UserName'],
                        picDir=os.path.join(download_dir, '%d.png' % image_id))
    return args


def get_chatroom_by_name(name, chatrooms):
    for chatroom in chatrooms:
        if chatroom['NickName'] == name:
            return chatroom


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="img", type=str, help="Folder to store the downloaded images")
    parser.add_argument("--type", type=str, choices=["self", "chatroom"], default="self")
    parser.add_argument("--name", type=str, help="Specify the chatroom name if type=chatroom")

    args = parser.parse_args()
    download_dir = args.dir

    if args.type == "self":
        print("Logging in...")
        itchat.auto_login(hotReload=True)

        print("Loading contact...")
        itchat.get_friends(update=True)
        friends = itchat.get_friends(update=True)
        download = download_friend

    elif args.type == "chatroom":
        assert len(args.name) > 0, "You must provide a chatroom name!"
        print("Logging in...")
        itchat.auto_login(hotReload=True)

        print("Getting chatrooms...")
        chatrooms = itchat.get_chatrooms(update=True)
        chatroom = get_chatroom_by_name(args.name, chatrooms)
        assert chatroom is not None, "Chatroom \"{}\" not found".format(args.name)

        print("Updating chatroom...")
        itchat.update_chatroom(chatroom['UserName'], True)

        # fetch the chatroom data again
        chatroom = get_chatroom_by_name(args.name, itchat.get_chatrooms())

        friends = chatroom['MemberList']
        download = download_chatroom_member
    else:
        raise Exception("Invalid argument")

    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)

    if os.path.isfile(os.path.join(download_dir, "cache.pkl")):
        downloaded = pickle.load(open(os.path.join(download_dir, "cache.pkl"), "rb"))
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
                f.append(pool.submit(download, (friend['UserName'], available_numbers[counter], download_dir)))
                counter += 1

        start_time = time.clock()
        for i, future in tqdm(enumerate(f), total=len(f), desc="[Downloading images]", unit="imgs"):
            try:
                if time.clock() - start_time > max_wait_time:
                    user_name, idx, _ = future.result(0)
                else:
                    user_name, idx, _ = future.result(ceil(max_wait_time - time.clock() + start_time))
                downloaded[user_name] = idx
            except concurrent.futures.TimeoutError:
                print("\nTimeout when downloading the head image of", friends[available_numbers[i]]['NickName'])

        if len(downloaded) < len(friends):
            print("Warning: Failed to download some of the images")
            print("Retrying...")

        pickle.dump(downloaded, open(os.path.join(download_dir, "cache.pkl"), "wb"))

    print("Success")
    exit(0)
