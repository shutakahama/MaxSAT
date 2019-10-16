#!/usr/bin/env python
# -*- coding: utf-8 -*-

#MaxSAT
import numpy as np
from pulp import *

#論理式の記述を簡単にするための関数
#入力xに対してtpが1ならx, 0なら(1-x)を返す
def tp(x,tp):
    if tp == 1:
        return x
    if tp == -1:
        return 1 - x

#randomized rounding
#入力xに対して確率xで1を返す
def rround(x):
    if np.random.rand() < x:
        return 1
    else:
        return 0

#論理式を表現する行列を生成する関数
#各変数xを真(1)にするか偽(-1)にするかをランダムに生成．
#2**k個のクローズ分生成し，重複があれば削除する
def make_mat(shape):
    A = np.random.choice((1,-1), shape)
    A = np.unique(A,axis=0)
    return A

#手法1
#各変数xを1/2の確率で真にし，代入して真になるクローズの個数を数える
def maxsat1(A):
    nx = len(A[0]) #変数の数
    nz = len(A) #クローズの数
    x = np.random.rand(nx) #各変数xを1/2の確率で真にする
    x = x > 0.5
    #print(x)

    # 1か0を設定した変数xを論理式に代入して真になるクローズの個数を数える
    real_value = 0
    for i in range(nz):
        zr = 0.0
        for j in range(nx):
            if tp(x[j],A[i,j]) == 1: #真になる変数が１つでもあればクローズの値vは真になる
                zr = 1.0
        real_value += zr
    #print('real z value: ', real_value)
    
    return real_value / nz

#手法1の実行関数
def func1(roop, k):
    alpha = 0
    # ランダムに生成した論理式Aに対してMaxSATを繰り返して評価する
    for i in range(roop):
        A = make_mat((2**k,k))
        #print('\n',A)
        _alpha = maxsat1(A)
        #print(_alpha)
        alpha += _alpha
    print('alpha: ', alpha/roop)



#手法2
#線形計画問題に緩和して解いたあとrandomized roundingをする手法
#変数xと各クローズを表現する変数zをおき，それぞれ0〜1までを動くとしてzの和を最大化するように最適化する．
#その後最適化したxをrandomized roundingで0か1に丸めて，　実際に真になるクローズの数を求める．

def maxsat2(A):
    nx = len(A[0]) #変数の数
    nz = len(A) #クローズの数
    x = []
    z = []
    # pulpを用いて最適化問題を解く
    m = LpProblem(sense=LpMaximize)
    
    # 変数x, z の定義，　扱いやすいように変数のリストを作る
    for i in range(nx):
        x.append(LpVariable('x{}'.format(i), lowBound=0, upBound=1))
    for i in range(nz):
        z.append(LpVariable('z{}'.format(i), lowBound=0, upBound=1))

    # 目的関数の定義, m = z1 + z2 + ... + zn
    m_ob = 0
    for i in range(nz):
        m_ob += z[i]
    m += m_ob
    
    # 条件式の定義, x1 + x2 + ... xn >= zj のような形
    for i in range(nz):
        m_c = 0
        for j in range(nx):
            m_c += tp(x[j],A[i,j])
        m += m_c >= z[i]
    
    # 最適化
    m.solve()
    
    #最適化後の各変数x,zのprint
    #for i in range(nx):
    #    print('x{}'.format(i), value(x[i]))
    #for i in range(nz):
    #    print('z{}'.format(i), value(z[i]))
        
    # 理想的な真のクローズの個数, 最適化された全てのzの和
    ideal_value = 0
    for i in range(nz):
        ideal_value += value(z[i])
    #print('ideal z value: ', ideal_value)
    
    # randomized rounding
    _x = []
    for i in range(nx):
        _x.append(rround(value(x[i])))
    #print(_x)
    
    # 実際の真のクローズの個数，　丸めた変数xを論理式に代入して真になるクローズの個数を数える
    real_value = 0
    for i in range(nz):
        zr = 0.0
        for j in range(nx):
            if tp(_x[j],A[i,j]) == 1: #真になる変数が１つでもあればクローズの値zrは真になる
                zr = 1.0
        real_value += zr
    #print('real z value: ', real_value)
    #print('nz: {}, ideal z value: {}, real z value: {},  '.format(nz, ideal_value, real_value))
    
    return real_value / ideal_value, real_value / nz

# 手法2の実行関数
def func2(roop, k):
    beta = 0
    gamma = 0
    # ランダムに生成した論理式Aに対してMaxSATを繰り返して評価する
    for i in range(roop):
        A = make_mat((2**k,k))
        #print('\n',A)
        _beta, _gamma = maxsat2(A)
        #print(_beta)
        beta += _beta
        gamma += _gamma
    print('beta: ', beta/roop)
    print('gamma: ', gamma/roop)



#ある特定の論理式Aに対してMaxSATを行いたい場合
def func3(roop, A):
    beta = 0
    gamma = 0
    for i in range(roop):
        _beta, _gamma = maxsat2(A)
        beta += _beta
        gamma += _gamma
    print('beta: ', beta/roop)
    print('gamma: ', gamma/roop)



#main
roop = 100 #繰り返しの回数
k = 2 #変数の数
#func1(roop,k)
func2(roop,k)

#A = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
#func3(roop,A)
