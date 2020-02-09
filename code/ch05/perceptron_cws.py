import zipfile
import os

from pyhanlp import *
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


sighan05 = ensure_data('icwb2-data', 'http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip')
msr_train = os.path.join(sighan05, 'training', 'msr_training.utf8')
msr_model = os.path.join(test_data_path(), 'msr_cws')
msr_test = os.path.join(sighan05, 'testing', 'msr_test.utf8')
msr_output = os.path.join(sighan05, 'testing', 'msr_bigram_output.txt')
msr_gold = os.path.join(sighan05, 'gold', 'msr_test_gold.utf8')
msr_dict = os.path.join(sighan05, 'gold', 'msr_training_words.utf8')

## ===============================================
## 以下开始中文分词



CWSTrainer = JClass('com.hankcs.hanlp.model.perceptron.CWSTrainer')
CWSEvaluator = SafeJClass('com.hankcs.hanlp.seg.common.CWSEvaluator')
HanLP.Config.ShowTermNature = False   # 关闭显示词性


def train_uncompressed_model():
    model = CWSTrainer().train(msr_train, msr_train, msr_model, 0., 10, 8).getModel()  # 训练模型
    model.save(msr_train, model.featureMap.entrySet(), 0, True)  # 最后一个参数指定导出txt


def train():
    model = CWSTrainer().train(msr_train, msr_model).getModel()  # 训练感知机模型
    segment = PerceptronLexicalAnalyzer(model).enableCustomDictionary(False)  # 创建感知机分词器
    print(CWSEvaluator.evaluate(segment, msr_test, msr_output, msr_gold, msr_dict))  # 标准化评测
    return segment
    


segment = train()
sents = [
    "王思斌，男，１９４９年１０月生。",
    "山东桓台县起凤镇穆寨村妇女穆玲英",
    "现为中国艺术研究院中国文化研究所研究员。",
    "我们的父母重男轻女",
    "北京输气管道工程",
]
for sent in sents:
    print(segment.seg(sent))
# train_uncompressed_model()