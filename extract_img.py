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
import unicodedata
import re
import traceback


def slugify(value, allow_unicode=True):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def download_pic(args):
    try:
        itchat.get_head_img(**args)
        return os.path.getsize(args['picDir'])
    except:
        traceback.print_exc()
        return 0


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
                "picDir": os.path.join(download_dir, slugify(f"{mem['NickName']}_{mem['RemarkName']}") + ".jpg")
            }

    if args.type == "groupchat" or args.type == "all":
        print("Getting groupchats...")
        chatrooms = itchat.get_chatrooms(update=True)
        # Nickname PYQuanPin RemarkPYQuanPin
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
            cr_name = chatroom['NickName']
            for mem in chatroom['MemberList']:
                if mem['UserName'] not in download_args:
                    download_args[mem['UserName']] = {
                        "userName": mem['UserName'],
                        'chatroomUserName': chatroom['UserName'],
                        "picDir": os.path.join(download_dir, slugify(f"{cr_name}_{mem['NickName']}") + ".jpg")
                    }

    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)

    download_args = download_args.values()
    download_args = [arg for arg in download_args if not os.path.exists(arg['picDir']) or os.path.getsize(arg['picDir']) == 0]
    
    pool = ThreadPoolExecutor(min(mp.cpu_count(), len(download_args)))
    pbar = tqdm(desc="[Downloading]", total=len(download_args))
    count = 1
    while len(download_args) > 0:
        download_args = [arg for arg in download_args if not os.path.exists(arg['picDir']) or os.path.getsize(arg['picDir']) == 0]
        try:
            for sz in pool.map(download_pic, download_args, timeout=20):
                if sz > 0:
                    pbar.update()
        except TimeoutError:
            pool.shutdown(False)
            pool = ThreadPoolExecutor(min(mp.cpu_count(), len(download_args)))
            tqdm.write(f"timeout downloading pics, retrying.... attempt {count}")
            count += 1
    pool.shutdown()