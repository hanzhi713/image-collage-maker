"""
extract_img.py
get profile pictures from your WeChat friends or group chat members
"""
import itchat
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm import tqdm
import argparse
import multiprocessing as mp


def download_pic(args):
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
                        choices=["self", "groupchat", "all"], default="self")
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

    download_args = dict()
    if args.type == "self" or args.type == "all":
        print("Loading contact...")
        for mem in itchat.get_friends(update=True):
            download_args[mem['UserName']] = {
                "userName": mem['UserName'], 
                "picDir": os.path.join(download_dir, f"{mem['UserName']}.jpg")
            }

    if args.type == "groupchat" or args.type == "all":
        print("Getting groupchats...")
        chatrooms = itchat.get_chatrooms(update=True)
        if not args.name:
            print("Updating groupchats... this might take a while (several minutes if you have tens of large group chats)")
            itchat.update_chatroom([chatroom['UserName'] for chatroom in chatrooms], True)
            chatrooms = itchat.get_chatrooms()
        else:
            filtered_chatrooms = []
            for name in args.name:
                chatroom = get_chatroom_by_name(args.name, chatrooms)
                assert chatroom is not None, f"Chatroom \"{args.name}\" not found"
                filtered_chatrooms.append(chatroom)
            
            itchat.update_chatroom([chatroom['UserName'] for chatroom in filtered_chatrooms], True)
            chatrooms = itchat.get_chatrooms()
            chatrooms = [get_chatroom_by_name(name, chatrooms) for name in args.name]
        
        for chatroom in chatrooms:
            for mem in chatroom['MemberList']:
                uname = mem['UserName']
                if uname not in download_args:
                    download_args[uname] = {
                        "userName": uname,
                        'chatroomUserName': chatroom['UserName'],
                        "picDir": os.path.join(download_dir, f"{uname}.jpg")
                    }

    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)

    download_args = download_args.values()
    download_args = [arg for arg in download_args if not os.path.exists(arg['picDir']) or os.path.getsize(arg['picDir']) == 0]
    
    pool = ThreadPoolExecutor(min(mp.cpu_count() * 4, len(download_args)))
    pbar = tqdm(desc="[Downloading]", total=len(download_args))
    count = 1
    while len(download_args) > 0:
        download_args = [arg for arg in download_args if not os.path.exists(arg['picDir'])]
        try:
            for _ in pool.map(download_pic, download_args, timeout=10):
                pbar.update()
        except TimeoutError:
            tqdm.write(f"timeout downloading pics, retrying.... attempt {count}")
            count += 1