## 加减乘除
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190711195108688.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
## 矩阵相乘
torch.mm 只适用于2d，不推荐
torch.matmul 推荐使用
@ 运算符重载
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071120060893.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
## 次方运算

    a = torch.full([2,2],3)
    aa = a.pow(2)
 

   操作运算如下：

`a.pow(n)`  表示a^n^
`a**n`  表示a^n^
`aa.sqrt()`  表示aa ^1/2^
`aa.rsqrt()`  表示aa^1/3^
    注意这里没有torch.

## 幂/对数

    a = torch.exp(torch.one(2,2)) #指数运算
    b = torch.log(a) #对数运算，默认底数为e
    c = torch.log2(a) #对数运算，改变底数

## 近似值
向下取整，floor()
向上取整，ceil()
取整数部分，trunc()
取小数部分，frac()
四舍五入法，round()
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190711202056683.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
## clamp()

 - clamp(min) 
 - clamp(min,max)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190711202657582.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
# 数据统计
## norm范数

    a.norm(n,dim=m) #n表示求n的范数，m表示第m维

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190711203928869.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)

## mean(),sum(),min(),max(),prod()累乘,argmax()返回最大值的下标,argmin()返回最小值的索引
argmax(dim=n)表示在第n维的最大值，如果不指定维度，则会打平为一维数据，返回所有数据的最大值的索引。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190711204813147.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
## dim,keepdim
keepdim表示维度信息和原来的一样
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190711205535691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
## Top-k/k-th

    a.topk(k,dim=n， largest=False) #k表示前k个，dim表示维度，largest表示最大值，
    默认True，要得到最小值就改为Flase
a.kthvalue(k,dim=n) #表示第n维的第k小的值
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190711210355611.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZXJhX3Fpbmc=,size_16,color_FFFFFF,t_70)
## 比较
`>,>=,<,<=,!=,==`
判断每一个元素是否相等`torch.eq(a,b)`，返回维度相同，值为0/1的矩阵
判断整个数据是否相等`torch.equal(a,b)`，返回True或者False

