# main.py
# 메인 실행 파일

from preproccessing import Preprocessor

def main():
    # feature preprocessing (from preproccessing.py)
    pp = Preprocessor() # 파라미터 2개 모두 기본값 사용
    pp.step1() # print(df1) # 잘됨
    print("step1 완료")
    df_merged = pp.step2(df_merged_path='./cache/df_merged_0025.parquet',
                         time='0.0025s',
                         dbc_path='datasets/hyundai_2015_ccan.dbc') # 모두 기본값 사용
    print(df_merged)

    # sliding window
    # TODO

    # model (from conv2d.py)
    # autoencoder


if __name__ == "__main__":
    result_df = main()