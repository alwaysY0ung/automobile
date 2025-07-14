# main.py
# 메인 실행 파일

from preprocessing import TrainPreprocessor
from preprocessing import TestPreprocessor

def main():
    # train datasets preprocessing
    # feature preprocessing (from preproccessing.py)
    train_pp1 = TrainPreprocessor(raw_path='./datasets/logfile2022-11-01_11-56-50.txt',
                 cache_path='./cache/df1.parquet') # 파라미터 2개 모두 기본값 사용
    train_pp2 = TrainPreprocessor(raw_path='./datasets/logfile2022-11-01_12-21-04.txt',
                 cache_path='./cache/df1.parquet')
    train_pp3 = TrainPreprocessor(raw_path='./datasets/logfile2022-11-01_13-00-02.txt',
                 cache_path='./cache/df1.parquet')
    train_pp4 = TrainPreprocessor(raw_path='./datasets/logfile2022-11-01_13-25-08.txt',
                 cache_path='./cache/df1.parquet')
    train_pp5 = TrainPreprocessor(raw_path='./datasets/logfile2022-11-01_14-02-06.txt',
                cache_path='./cache/df1.parquet')

    train_pp1.step1() # print(df1) # 잘됨
    train_pp2.step1()
    train_pp3.step1()
    train_pp4.step1()
    train_pp5.step1()
    print(f"{__name__}: train datasets, step1 완료")

    df1_merged = train_pp1.step2(df_merged_path='./cache/df1_merged_005.parquet',
                         time='0.005s',
                         dbc_path='datasets/hyundai_2015_ccan.dbc') # 모두 기본값 사용

    df2_merged = train_pp2.step2(df_merged_path='./cache/df2_merged_005.parquet',
                         time='0.005s',
                         dbc_path='datasets/hyundai_2015_ccan.dbc') 

    df3_merged = train_pp2.step2(df_merged_path='./cache/df3_merged_005.parquet',
                        time='0.005s',
                        dbc_path='datasets/hyundai_2015_ccan.dbc') 

    df4_merged = train_pp2.step2(df_merged_path='./cache/df4_merged_005.parquet',
                        time='0.005s',
                        dbc_path='datasets/hyundai_2015_ccan.dbc')
    
    df5_merged = train_pp2.step2(df_merged_path='./cache/df5_merged_005.parquet',
                        time='0.005s',
                        dbc_path='datasets/hyundai_2015_ccan.dbc')


    print(f"{__name__}: train datasets, step2 완료")

    # test datasets preprocessing
    # test_pp = TestPreprocessor(raw_data_dir='datasets/test')
    test_pp = TestPreprocessor(raw_data_dir='datasets/intrusion_datasets')
    test_pp.step1(time='0.005s')
    print(f"{__name__}: test datasets, step1 완료")

    # model (from conv2d.py)
    # autoencoder


if __name__ == "__main__":
    result_df = main()