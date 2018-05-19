# -*- coding: utf-8 -*-

# 保存原始测试集instance顺序

import pandas as pd

save_path = 'result/initial_instance_id.csv'

df = pd.read_csv('data/origin/round2_ijcai_18_test_b_20180510.csv')

df.to_csv(save_path,columns=['instance_id'],index=False) 

print('原始instance_id保存完成')