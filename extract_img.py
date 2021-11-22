"""
extract_img.py
get profile pictures from your WeChat friends or group chat members
"""
import itchat
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import multiprocessing as mp


def download_pic(args):
    fpath = args['picDir']
    if not os.path.exists(fpath):
        itchat.get_head_img(**args)


def get_chatroom_by_name(name, chatrooms):
    for chatroom in chatrooms:
        if chatroom['NickName'] == name:
            return chatroom


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="img", type=str,
                        help="Folder to store the downloaded images")
    parser.add_argument("--type", type=str,
                        choices=["self", "groupchat"], default="self")
    parser.add_argument("--name", type=str, nargs="+",
                        help="Specify the chatroom name if type=chatroom")
    parser.add_argument("--list_chatroom", action="store_true")

    args = parser.parse_args()
    download_dir = args.dir

    print("Logging in...")
    itchat.auto_login(hotReload=True)

    if args.list_chatroom:
        print("Getting chatrooms...")
        chatrooms = itchat.get_chatrooms(update=True)
        for chatroom in tqdm(chatrooms, desc="[Updating Groupchats]"):
            tqdm.write(chatroom['NickName'])
        exit()

    if args.type == "self":
        print("Loading contact...")
        download_args = [
            {
                "userName": mem['UserName'], 
                "picDir": os.path.join(download_dir, f"{mem['UserName']}.jpg")
            } 
                for mem in itchat.get_friends(update=True)
        ]

    elif args.type == "groupchat":
        print("Getting groupchats...")
        chatrooms = itchat.get_chatrooms(update=True)
        if not args.name:
            for chatroom in tqdm(chatrooms, desc="[Updating Groupchats]"):
                itchat.update_chatroom(chatroom['UserName'], True)
            chatrooms = itchat.get_chatrooms()
        else:
            filtered_chatrooms = []
            for name in tqdm(args.name, desc="[Updating Groupchats]"):
                chatroom = get_chatroom_by_name(args.name, chatrooms)
                assert chatroom is not None, f"Chatroom \"{args.name}\" not found"

                itchat.update_chatroom(chatroom['UserName'], True)

                # fetch the chatroom data again
                filtered_chatrooms.append(get_chatroom_by_name(args.name, itchat.get_chatrooms()))
            chatrooms = filtered_chatrooms
        
        download_args = []
        for chatroom in chatrooms:
            for mem in chatroom['MemberList']:
                download_args.append(
                    {
                        "userName": mem['UserName'],
                        'chatroomUserName': chatroom['UserName'],
                        "picDir": os.path.join(download_dir, f"{mem['UserName']}.jpg")
                    }
                )
    else:
        raise Exception("Invalid argument")

    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)

    pool = ThreadPoolExecutor(max(mp.cpu_count() * 4, len(download_args)))
    for _ in tqdm(pool.map(download_pic, download_args, chunksize=32), desc="[Downloading]", total=len(download_args)):
        pass
