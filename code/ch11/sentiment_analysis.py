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



## 验证是否存在语料库，如果没有自动下载
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


# 中文情感挖掘语料-ChnSentiCorp 谭松波
chn_senti_corp = ensure_data("ChnSentiCorp情感分析酒店评论", "http://file.hankcs.com/corpus/ChnSentiCorp.zip")


## ===============================================
## 以下开始 情感分析


IClassifier = JClass('com.hankcs.hanlp.classification.classifiers.IClassifier')
NaiveBayesClassifier = JClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')



def predict(classifier, text):
    print("《%s》 情感极性是 【%s】" % (text, classifier.classify(text)))


if __name__ == '__main__':
    classifier = NaiveBayesClassifier()
    #  创建分类器，更高级的功能请参考IClassifier的接口定义
    classifier.train(chn_senti_corp)
    #  训练后的模型支持持久化，下次就不必训练了
    predict(classifier, "前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！")
    predict(classifier, "结果大失所望，灯光昏暗，空间极其狭小，床垫质量恶劣，房间还伴着一股霉味。")
    predict(classifier, "可利用文本分类实现情感分析，效果不是不行")