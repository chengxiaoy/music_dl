import requests
from bs4 import BeautifulSoup
import urllib.request
from lxml import html
import sys
import os

# etree = html.etree
# 这里是设置请求头
headers = {
    'Referer': 'http://music.163.com/',
    'Host': 'music.163.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
}


def download_playlist(playlist_id, category):
    # 歌单的url地址这里改id
    play_url = 'http://music.163.com/playlist?id=' + str(playlist_id)

    s = requests.session()
    response = s.get(play_url, headers=headers).content

    s = BeautifulSoup(response, 'lxml')
    main = s.find('ul', {'class': 'f-hide'})
    print(main.find_all('a'))
    lists = []
    for music in main.find_all('a'):
        list = []
        # print('{} : {}'.format(music.text, music['href']))
        musicUrl = 'http://music.163.com/song/media/outer/url' + music['href'][5:] + '.mp3'
        musicName = music.text
        # 单首歌曲的名字和地址放在list列表中
        list.append(musicName)
        list.append(musicUrl)
        # 全部歌曲信息放在lists列表中
        lists.append(list)

    data_dir = './audio/' + category + '/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for i in lists:
        url = i[1]
        name = i[0]
        try:
            print('正在下载', name)
            # 这里修改路径，随便指定盘符，但是得存在
            urllib.request.urlretrieve(url, data_dir + '%s.mp3' % name)
            print('下载成功')
        except Exception as e:
            print(e)
            print('下载失败')


if __name__ == '__main__':
    download_playlist(2916766519, '轻音乐')
    download_playlist(959196950, '轻音乐')
    download_playlist(2094091970, '布鲁斯')
    download_playlist(6708648, '布鲁斯')

