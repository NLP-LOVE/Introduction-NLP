

from  pyhanlp import *
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


## 指定 PKU 语料库
PKU98 = ensure_data("pku98", "http://file.hankcs.com/corpus/pku98.zip")
PKU199801 = os.path.join(PKU98, '199801.txt')
PKU199801_TRAIN = os.path.join(PKU98, '199801-train.txt')
PKU199801_TEST = os.path.join(PKU98, '199801-test.txt')
POS_MODEL = os.path.join(PKU98, 'pos.bin')
NER_MODEL = os.path.join(PKU98, 'ner.bin')


## ===============================================
## 以下开始 CRF 词性标注

AbstractLexicalAnalyzer = JClass('com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer')
PerceptronSegmenter = JClass('com.hankcs.hanlp.model.perceptron.PerceptronSegmenter')
CRFPOSTagger = JClass('com.hankcs.hanlp.model.crf.CRFPOSTagger')



def train_crf_pos(corpus):
    # 选项1.使用HanLP的Java API训练，慢
    tagger = CRFPOSTagger(None)  # 创建空白标注器
    tagger.train(corpus, POS_MODEL)  # 训练
    tagger = CRFPOSTagger(POS_MODEL) # 加载
    # 选项2.使用CRF++训练，HanLP加载。（训练命令由选项1给出）
    # tagger = CRFPOSTagger(POS_MODEL + ".txt")
    analyzer = AbstractLexicalAnalyzer(PerceptronSegmenter(), tagger)  # 构造词法分析器，与感知机分词器结合，能同时进行分词和词性标注。
    print(analyzer.analyze("李狗蛋的希望是希望上学"))  # 分词+词性标注
    print(analyzer.analyze("李狗蛋的希望是希望上学").translateLabels())  # 对词性进行翻译
    return tagger


if __name__ == '__main__':
    tagger = train_crf_pos(PKU199801_TRAIN)
