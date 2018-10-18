'''
extract_img.py
get images from WeChat friends list.
'''
import itchat

# 登陆
itchat.auto_login(hotReload=True)

# 获取通讯录列表
friends = itchat.get_friends(update=True)

# 编列列表中的字典，通过微信的用户唯一标识符 UserName 下载头像，头像的链接在 HeadImageUrl 中
for i, friend in enumerate(friends):
    itchat.get_head_img(userName=friend['UserName'],
                        picDir='img/%d.png' % i)
