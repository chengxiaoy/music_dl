import requests
from bs4 import BeautifulSoup
import urllib.request
from lxml import html
import sys
import os
from multiprocessing import Pool

from redis import Redis

# etree = html.etree
# 这里是设置请求头
headers = {
    'Referer': 'http://music.163.com/',
    'Host': 'music.163.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
}


def dowload_song(song_id):

    try:
        song_url = 'http://music.163.com/song/media/outer/url?id=' + str(song_id) + '.mp3'

        print('正在下载', song_id)
        # 这里修改路径，随便指定盘符，但是得存在

        subdir = 'data' + song_id // 20000

        data_dir = './audio/' + subdir + '/'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        urllib.request.urlretrieve(song_url, data_dir + '%s.mp3' % song_id)
        print('下载成功')
    except Exception as e:
        print(e)
        print('下载失败')


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
        list.append(music['href'][9:])
        # 全部歌曲信息放在lists列表中
        lists.append(list)

    data_dir = './audio/' + category + '/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for i in lists:
        url = i[1]
        name = i[0]
        song_id = i[2]
        try:
            print('正在下载', name)
            dowload_song(int(song_id))
            print('下载成功')
        except Exception as e:
            print(e)
            print('下载失败')


if __name__ == '__main__':
    conn = Redis(host='116.62.226.241', port=6379, password=154615)
    song_ids = conn.smembers('song_id')
    song_ids = [int(x) for x in song_ids]

    p = Pool(4)

    p.map(dowload_song, song_ids)

    # dowload_song(27501701)
    #
    # download_playlist(2916766519, '轻音乐')
# download_playlist(959196950, '轻音乐')
# download_playlist(2094091970, '布鲁斯')
# download_playlist(6708648, '布鲁斯')
# download_playlist(116523888, '华语流行')
# download_playlist(46140598, '华语流行')
# download_playlist(728056493, '华语流行')
# download_playlist(3812895, '电子舞曲')
# download_playlist(964308842, '电子舞曲')
# download_playlist(902526449, '电子舞曲')
# download_playlist(2012592880, '电子舞曲')
# download_playlist(2571885518, '粤语')
# download_playlist(2503316602, '粤语')
# download_playlist(755965175, '粤语')
# download_playlist(632021463, '粤语')
# download_playlist(2946227728, '民谣')
# download_playlist(329641066, '民谣')
# download_playlist(498708023, '民谣')
# download_playlist(2770902965, '民谣')
# download_playlist(995108392, '摇滚')
# download_playlist(934792457, '摇滚')
# download_playlist(521806549, '摇滚')
# download_playlist(250819, '摇滚')
# download_playlist(513217941, '后摇')
# download_playlist(11917099, '后摇')
# download_playlist(724206994, '后摇')
# download_playlist(509265479, '后摇')
# download_playlist(644883057, '古风')
# download_playlist(743558697, '古风')
# download_playlist(2889677087, '古风')
# download_playlist(43796081, '乡村')
# download_playlist(23217655, '乡村')
# download_playlist(89272476, '乡村')
# download_playlist(2968950064, '乡村')
