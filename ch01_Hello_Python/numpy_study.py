import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

tmp_arange = np.arange(16, dtype=np.int64)

reshaped_arange = tmp_arange.reshape(2, 2, 4)
print(reshaped_arange)
print(reshaped_arange.max(axis=0))


rng = np.random.default_rng()
samples = rng.normal(size=2500)
print(samples)


# numpy 연산과 for_loop 속도 비교
# %%
import numpy as np

# %%
# 여기 있는 코드를 수십 번 돌려서 평균 시간을 내줍니다.
a = np.arange(5)
b = np.arange(5, 10)
# %%
# %%timeit
c = []
for i in range(len(a)):
    c.append(a[i] * b[i])
c

# %%
# %%timeit
c = a * b
c
# %%

# 벡터화를 통해서 더 간단하게 처리 가능
# 간단하기 때문에 오류가 발생할 가능성이 매우 적으며
# 뒷단은 c언어와 포트란으로 작동하기에 매우 빠름


# numpy의 벡터 생성 방법

print(np.zeros(10))
print(np.ones((3, 3)))
x = np.full((3, 3), -1)
print(x[0][0])

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

for i in range(4):
    print(dx[i] + " : " + dy[i])
