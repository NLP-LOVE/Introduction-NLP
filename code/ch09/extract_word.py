
from pyhanlp import *



import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH

def test_data_path():
    """
    获取测试数据路径，位于$root/data/test，根目录由配置文件指定。
    :return:
    """
    data_path = os.path.join(HANLP_DATA_PATH, 'test')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    return data_path



## 验证是否存在 MSR语料库，如果没有自动下载
def ensure_data(data_name, data_url):
    root_path = test_data_path()
    dest_path = os.path.join(root_path, data_name)
    if os.path.exists(dest_path):
        return dest_path
    
    if data_url.endswith('.zip'):
        dest_path += '.zip'
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with zipfile.ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path[:-len('.zip')]
    return dest_path




HLM_PATH = ensure_data("红楼梦.txt", "http://file.hankcs.com/corpus/红楼梦.zip")
XYJ_PATH = ensure_data("西游记.txt", "http://file.hankcs.com/corpus/西游记.zip")
SHZ_PATH = ensure_data("水浒传.txt", "http://file.hankcs.com/corpus/水浒传.zip")
SAN_PATH = ensure_data("三国演义.txt", "http://file.hankcs.com/corpus/三国演义.zip")
WEIBO_PATH = ensure_data("weibo-classification", "http://file.hankcs.com/corpus/weibo-classification.zip")


def test_weibo():
    for folder in os.listdir(WEIBO_PATH):
        print(folder)
        big_text = ""
        for file in os.listdir(os.path.join(WEIBO_PATH, folder)):
            with open(os.path.join(WEIBO_PATH, folder, file)) as src:
                big_text += "".join(src.readlines())
        word_info_list = HanLP.extractWords(big_text, 100)
        print(word_info_list)


def extract(corpus):
    print("%s 热词" % corpus)
    
    ## 参数如下
    # reader: 文本数据源
    # size: 控制返回多少个词
    # newWordsOnly: 为真时，程序将使用内部词库过滤掉“旧词”。
    # max_word_len: 控制识别结果中最长的词语长度
    # min_freq: 控制结果中词语的最低频率
    # min_entropy: 控制结果中词语的最低信息熵的值，一般取 0.5 左右,值越大，越短的词语就越容易提取
    # min_aggregation: 控制结果中词语的最低互信息值，一般取 50 到 200.值越大，越长的词语越容易提取
    word_info_list = HanLP.extractWords(IOUtil.newBufferedReader(corpus), 100)
    print(word_info_list)
    # print("%s 新词" % corpus)
    # word_info_list = HanLP.extractWords(IOUtil.newBufferedReader(corpus), 100, True)
    # print(word_info_list)


if __name__ == '__main__':
    extract(HLM_PATH)
    extract(XYJ_PATH)
    extract(SHZ_PATH)
    extract(SAN_PATH)
    test_weibo()

    # 更多参数
    word_info_list = HanLP.extractWords(IOUtil.newBufferedReader(HLM_PATH), 100, True, 4, 0.0, .5, 100)
    print(word_info_list)

