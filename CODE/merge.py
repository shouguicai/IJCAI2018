# -*- coding: utf-8 -*-

# 整合各部分特征，作为训练数据

import numpy as np
import pandas as pd 
from multiprocessing import Process
import multiprocessing
import time

def merge(data_type='train'):
	df_output=''
	save_path=''
	if data_type=='train':
		df_comm =pd.read_csv('data/cutData/train/train_20180425_comm.csv')
		df_click =pd.read_csv('data/cutData/train/train_20180425_click.csv')
		df_zhl =pd.read_csv('data/cutData/train/train_20180425_zhl.csv')
		df_top3 =pd.read_csv('data/cutData/train/train_20180425_top3.csv')
		#df_time =pd.read_csv('data/cutData/train/train_20180425_time.csv')
		df_re_time =pd.read_csv('data/cutData/train/train_20180425_reverse_time.csv')
		df_snow =pd.read_csv('data/feature/snowDig.csv')
		df_search = pd.read_csv('data/feature/searchDig.csv')
		save_path='data/cutData/train/train_20180425_merge.csv'
	elif data_type=='test':
		df_comm =pd.read_csv('data/cutData/test/test_20180425_comm.csv')
		df_click =pd.read_csv('data/cutData/test/test_20180425_click.csv')
		df_zhl =pd.read_csv('data/cutData/test/test_20180425_zhl.csv')
		df_top3 =pd.read_csv('data/cutData/test/test_20180425_top3.csv')
		#df_time =pd.read_csv('data/cutData/test/test_20180425_time.csv')
		df_re_time =pd.read_csv('data/cutData/test/test_20180425_reverse_time.csv')
		df_snow =pd.read_csv('data/feature/snowDig.csv')
		df_search = pd.read_csv('data/feature/searchDig.csv')
		save_path='data/cutData/test/test_20180425_merge.csv'
	elif data_type=='validate':
		df_comm =pd.read_csv('data/cutData/validate/validate_20180425_comm.csv')
		df_click =pd.read_csv('data/cutData/validate/validate_20180425_click.csv')
		df_zhl =pd.read_csv('data/cutData/validate/validate_20180425_zhl.csv')
		df_top3 =pd.read_csv('data/cutData/validate/validate_20180425_top3.csv')
		#df_time =pd.read_csv('data/cutData/validate/validate_20180425_time.csv')
		df_re_time =pd.read_csv('data/cutData/validate/validate_20180425_reverse_time.csv')
		df_snow =pd.read_csv('data/feature/snowDig.csv')
		df_search = pd.read_csv('data/feature/searchDig.csv')
		save_path='data/cutData/validate/validate_20180425_merge.csv'
	else:
		print('data_type出错！')
		return
		
	df_output = df_comm
	del df_comm
	# .drop(['PH_item','PH_shop'],axis=1)
	df_output = pd.merge(df_output,df_zhl,how='left',on=['instance_id','item_id','user_id','context_id','shop_id'])
	del df_zhl
	df_output = pd.merge(df_output,df_click,how='left',on=['instance_id','item_id','user_id','context_id','shop_id'])
	del df_click
	df_output = pd.merge(df_output,df_top3,how='left',on=['instance_id','item_id','user_id','context_id','shop_id'])
	del df_top3
	#df_output = pd.merge(df_output,df_time,how='left',on=['instance_id','item_id','user_id','context_id','shop_id'])
	df_output = pd.merge(df_output,df_re_time,how='left',on=['instance_id','item_id','user_id','context_id','shop_id'])
	del df_re_time
	df_output = pd.merge(df_output,df_snow,how='left',on=['instance_id'])
	del df_snow
	df_output = pd.merge(df_output,df_search,how='left',on=['instance_id','item_id','user_id','context_id','shop_id'])
	del df_search
	
	df_output.to_csv(save_path,index=False) 
	
	print('保存完成')

if __name__=='__main__':
	p1 = Process(target=merge, args=('train',))
	p2 = Process(target=merge, args=('validate',))
	p3 = Process(target=merge, args=('test',))
	start = time.time()
	p1.start()
	p2.start()
	p3.start()
	p1.join()
	p2.join()
	p3.join()
	end = time.time()
	print('merge finished ... 3 processes take %s seconds' % (end - start))