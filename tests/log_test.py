from src.log import write_log, read_log

def test_ingest_log_isolated(tmp_path): write_log('ingest',{'rows':1},tmp_path); assert len(read_log('ingest',tmp_path))==1

def test_train_log_isolated(tmp_path): write_log('train',{'model':'x'},tmp_path); assert read_log('train',tmp_path)[0]['event']['model']=='x'

def test_predict_log_isolated(tmp_path): write_log('predict',{'latency_ms':1},tmp_path); assert read_log('predict',tmp_path)[0]['event']['latency_ms']==1

def test_invalid_log_kind(tmp_path):
    try: write_log('bad',{},tmp_path); assert False
    except ValueError: assert True
