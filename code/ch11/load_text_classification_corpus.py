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


sogou_corpus_path = ensure_data('搜狗文本分类语料库迷你版', 'http://file.hankcs.com/corpus/sogou-text-classification-corpus-mini.zip')




AbstractDataSet = JClass('com.hankcs.hanlp.classification.corpus.AbstractDataSet')
Document = JClass('com.hankcs.hanlp.classification.corpus.Document')
FileDataSet = JClass('com.hankcs.hanlp.classification.corpus.FileDataSet')
MemoryDataSet = JClass('com.hankcs.hanlp.classification.corpus.MemoryDataSet')

# 演示加载文本分类语料库
if __name__ == '__main__':
    dataSet = MemoryDataSet()  # ①将数据集加载到内存中
    dataSet.load(sogou_corpus_path)  # ②加载data/test/搜狗文本分类语料库迷你版
    dataSet.add("自然语言处理", "自然语言处理很有趣")  # ③新增样本
    allClasses = dataSet.getCatalog().getCategories()  # ④获取标注集
    print("标注集：%s" % (allClasses))
    for document in dataSet.iterator():
        print("第一篇文档的类别：" + allClasses.get(document.category))
        break